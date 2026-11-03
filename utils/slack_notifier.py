"""Slack webhook notifier for training progress.

Emits event-based messages (start / epoch / best / end / error) to a Slack
incoming webhook. All calls are best-effort: notifier failures never propagate
to the training loop, and non-master DDP ranks become no-ops automatically.

Enable via the ``slack`` section in a training YAML:

    slack:
      enabled: true
      webhook_url: "<SLACK_WEBHOOK_URL>"  # or set SLACK_WEBHOOK_URL env var
      notify_on:
        train_start: true
        epoch_end: true
        best_model: true
        train_end: true
        train_error: true
      progress_milestones: [0, 25, 50, 75, 100]  # % of total epochs
      epoch_interval: 1  # legacy fallback when progress_milestones is null
      mention: null  # e.g. "<!channel>" or "<@U12345>"
      timeout_sec: 5
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from omegaconf import DictConfig, OmegaConf

from utils.distributed_utils import is_master_process

LOGGER = logging.getLogger(__name__)

PLACEHOLDER_URL = "<SLACK_WEBHOOK_URL>"
ENV_WEBHOOK_URL = "SLACK_WEBHOOK_URL"
KST = timezone(timedelta(hours=9), name="KST")

DEFAULT_NOTIFY_ON = {
    "train_start": True,
    "epoch_end": True,
    "best_model": True,
    "train_end": True,
    "train_error": True,
}


class SlackNotifier:
    """Post training lifecycle events to a Slack incoming webhook.

    Attributes:
        webhook_url (str | None): Resolved webhook URL. ``None`` disables posts.
        run_name (str): Training run identifier included in every message.
        notify_on (dict[str, bool]): Per-event toggles.
        progress_milestones (list[int] | None): Progress percentages (0-100) at
            which to post ``on_epoch_end``. Takes precedence over
            ``epoch_interval`` when set. ``None`` falls back to
            ``epoch_interval``.
        epoch_interval (int): Legacy fallback — post ``on_epoch_end`` every N
            epochs when ``progress_milestones`` is not set.
        mention (str | None): Slack mention string prepended to messages.
        timeout_sec (float): HTTP timeout for webhook POST.
        hostname (str): Machine hostname for context.
    """

    def __init__(
        self,
        webhook_url: str | None,
        run_name: str,
        *,
        notify_on: dict[str, bool] | None = None,
        progress_milestones: list[int] | None = None,
        epoch_interval: int = 1,
        mention: str | None = None,
        timeout_sec: float = 5.0,
    ) -> None:
        """Initialize the notifier.

        Args:
            webhook_url (str | None): Slack incoming webhook URL. ``None`` or the
                placeholder string disables posting.
            run_name (str): Training run name.
            notify_on (dict[str, bool] | None): Per-event enable flags.
                Unspecified events fall back to defaults.
            progress_milestones (list[int] | None): Progress percentages (0-100)
                at which ``on_epoch_end`` should post. Values outside [0, 100]
                are discarded. When set, overrides ``epoch_interval``.
            epoch_interval (int): Emit ``on_epoch_end`` every N epochs. Used
                only when ``progress_milestones`` is None.
            mention (str | None): Optional Slack mention (e.g. ``"<!channel>"``).
            timeout_sec (float): HTTP request timeout in seconds.
        """
        self.webhook_url = webhook_url if _is_valid_webhook(webhook_url) else None
        self.run_name = run_name
        merged: dict[str, bool] = dict(DEFAULT_NOTIFY_ON)
        if notify_on:
            merged.update({k: bool(v) for k, v in notify_on.items()})
        self.notify_on = merged
        self.progress_milestones: list[int] | None
        if progress_milestones is None:
            self.progress_milestones = None
        else:
            cleaned = sorted(
                {int(m) for m in progress_milestones if 0 <= int(m) <= 100}
            )
            self.progress_milestones = cleaned or None
        self.epoch_interval = max(1, int(epoch_interval))
        self.mention = mention
        self.timeout_sec = float(timeout_sec)
        self.hostname = socket.gethostname()

    @classmethod
    def from_config(cls, cfg: DictConfig, run_name: str) -> "SlackNotifier":
        """Build a notifier from a top-level training config.

        Reads the ``slack`` section if present. When ``slack.enabled`` is false
        or missing, returns a disabled notifier (all methods become no-ops).
        Falls back to the ``SLACK_WEBHOOK_URL`` environment variable when the
        config value is the placeholder string or empty.

        Args:
            cfg (DictConfig): Full training config (OmegaConf).
            run_name (str): Training run name.

        Returns:
            SlackNotifier: Configured notifier instance.
        """
        # A bring-up smoke run uses the real training config, so it would post
        # the real config's alerts -- and a time-boxed run always ends on
        # SIGINT, which reaches the channel as "[FAILED] KeyboardInterrupt" for
        # a run that did exactly what it was asked to. False alarms on a channel
        # people watch for real failures are worse than no alarms, so a caller
        # that knows its run is disposable can say so.
        if os.environ.get("SPLICING_SLACK_DISABLE", "") not in ("", "0"):
            return cls(webhook_url=None, run_name=run_name)

        slack_cfg = OmegaConf.select(cfg, "slack")
        if slack_cfg is None or not bool(OmegaConf.select(cfg, "slack.enabled")):
            return cls(webhook_url=None, run_name=run_name)

        raw_url = OmegaConf.select(cfg, "slack.webhook_url")
        resolved_url = _resolve_webhook_url(raw_url)

        notify_on_cfg = OmegaConf.select(cfg, "slack.notify_on")
        if notify_on_cfg is None:
            notify_on: dict[str, bool] | None = None
        else:
            notify_on = {k: bool(v) for k, v in dict(notify_on_cfg).items()}

        milestones_cfg = OmegaConf.select(cfg, "slack.progress_milestones")
        if milestones_cfg is None:
            progress_milestones: list[int] | None = None
        else:
            progress_milestones = [int(m) for m in list(milestones_cfg)]

        return cls(
            webhook_url=resolved_url,
            run_name=run_name,
            notify_on=notify_on,
            progress_milestones=progress_milestones,
            epoch_interval=int(OmegaConf.select(cfg, "slack.epoch_interval") or 1),
            mention=OmegaConf.select(cfg, "slack.mention"),
            timeout_sec=float(OmegaConf.select(cfg, "slack.timeout_sec") or 5.0),
        )

    @property
    def is_active(self) -> bool:
        """Return True when at least one event can be posted.

        Returns:
            bool: True if the webhook is configured and the current process is
                the master rank.
        """
        return self.webhook_url is not None and is_master_process()

    def on_train_start(self, summary: dict[str, Any]) -> None:
        """Post a training-start message with configuration summary.

        Args:
            summary (dict[str, Any]): Key fields to render (e.g. model variant,
                epochs, batch size, world size, dataset path).
        """
        if not self._should_notify("train_start"):
            return
        lines = [f"*[START] Training started* — `{self.run_name}`"]
        lines.extend(_format_fields(summary))
        self._post("\n".join(lines))

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_metrics: dict[str, float] | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Post an epoch-completion message with training/validation metrics.

        Gating logic:
          * If ``progress_milestones`` is set, posts only when the current
            epoch matches one of the milestone target epochs computed from
            ``total_epochs``.
          * Otherwise falls back to ``epoch_interval`` modulo throttling.
          * The final epoch (``epoch + 1 == total_epochs``) is always posted.

        Args:
            epoch (int): Zero-based epoch index that just completed.
            total_epochs (int): Total configured epochs.
            train_loss (float): Average training loss for the epoch.
            val_metrics (dict[str, float] | None): Validation metrics to render.
            learning_rate (float | None): Current learning rate.
        """
        if not self._should_notify("epoch_end"):
            return
        one_based = epoch + 1
        if not self._is_epoch_reportable(one_based, total_epochs):
            return

        pct = int(round(100 * one_based / total_epochs)) if total_epochs > 0 else 0
        fields: dict[str, Any] = {
            "epoch": f"{one_based}/{total_epochs} ({pct}%)",
            "train_loss": f"{train_loss:.6f}",
        }
        if learning_rate is not None:
            fields["learning_rate"] = f"{learning_rate:.2e}"
        if val_metrics:
            for key, value in _select_summary_metrics(val_metrics).items():
                fields[f"val_{key}"] = _format_scalar(value)

        header = (
            f"*[EPOCH] Epoch {one_based}/{total_epochs} ({pct}%)* — `{self.run_name}`"
        )
        lines = [header]
        lines.extend(_format_fields(fields))
        self._post("\n".join(lines))

    def _is_epoch_reportable(self, one_based_epoch: int, total_epochs: int) -> bool:
        """Return True when the given epoch should trigger an epoch_end post.

        Args:
            one_based_epoch (int): 1-based epoch index that just completed.
            total_epochs (int): Total configured epochs.

        Returns:
            bool: True when the epoch matches milestones or interval criteria,
                or when it is the final epoch.
        """
        is_last = one_based_epoch >= total_epochs
        if is_last:
            return True
        if self.progress_milestones is not None:
            targets = _milestone_target_epochs(self.progress_milestones, total_epochs)
            return one_based_epoch in targets
        return one_based_epoch % self.epoch_interval == 0

    def on_best_model(
        self,
        epoch: int,
        best_val_loss: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Post a best-model-updated message.

        Args:
            epoch (int): Zero-based epoch index when the best model was found.
            best_val_loss (float): New best validation loss.
            extra (dict[str, Any] | None): Additional fields to render.
        """
        if not self._should_notify("best_model"):
            return
        fields: dict[str, Any] = {
            "epoch": epoch + 1,
            "best_val_loss": f"{best_val_loss:.6f}",
        }
        if extra:
            fields.update(extra)
        lines = [f"*[BEST] New best model* — `{self.run_name}`"]
        lines.extend(_format_fields(fields))
        self._post("\n".join(lines))

    def on_train_end(
        self,
        status: str,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Post a training-end message.

        Args:
            status (str): ``"completed"`` / ``"early_stopped"`` / ``"failed"``.
            summary (dict[str, Any] | None): Final metrics / paths to render.
        """
        if not self._should_notify("train_end"):
            return
        marker = "[DONE]" if status != "failed" else "[FAILED]"
        lines = [
            f"*{marker} Training {status}* — `{self.run_name}`",
        ]
        if summary:
            lines.extend(_format_fields(summary))
        self._post("\n".join(lines))

    def on_train_error(self, exception: BaseException) -> None:
        """Post a training-error message.

        Args:
            exception (BaseException): The uncaught exception that stopped training.
        """
        if not self._should_notify("train_error"):
            return
        fields = {
            "error_type": type(exception).__name__,
            "error_message": str(exception)[:500],
        }
        lines = [
            f"*[FAILED] Training failed* — `{self.run_name}`",
        ]
        lines.extend(_format_fields(fields))
        self._post("\n".join(lines))

    def _should_notify(self, event: str) -> bool:
        """Gate a single event on master rank, webhook presence, and config toggle.

        Args:
            event (str): Event key in ``notify_on``.

        Returns:
            bool: True when the event should be posted.
        """
        if not self.is_active:
            return False
        return bool(self.notify_on.get(event, False))

    def _post(self, text: str) -> None:
        """POST a text payload to the Slack webhook.

        Args:
            text (str): Message body (Slack mrkdwn).
        """
        if self.webhook_url is None:
            return

        prefix = f"{self.mention} " if self.mention else ""
        now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z")
        footer = f"_host: `{self.hostname}` · {now_str}_"
        payload = {"text": f"{prefix}{text}\n{footer}"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_sec) as resp:
                if resp.status >= 400:
                    LOGGER.warning("Slack webhook returned status %s", resp.status)
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            LOGGER.warning("Slack webhook POST failed: %s", exc)


def _milestone_target_epochs(milestones: list[int], total_epochs: int) -> set[int]:
    """Map milestone percentages to 1-based target epoch indices.

    Each percentage ``p`` maps to ``max(1, ceil(total_epochs * p / 100))``.
    Using ceil ensures the alert fires at or after the requested progress mark
    (e.g. 25% of 50 epochs = 12.5 → epoch 13). The 100% milestone always
    resolves to ``total_epochs``. When multiple milestones collapse to the same
    epoch (small ``total_epochs``) duplicates are dropped.

    Args:
        milestones (list[int]): Progress percentages (0-100), already sorted
            and deduped.
        total_epochs (int): Total configured epochs (must be > 0).

    Returns:
        set[int]: 1-based epoch indices that correspond to milestones.
    """
    if total_epochs <= 0:
        return set()
    targets: set[int] = set()
    for pct in milestones:
        target = max(1, math.ceil(total_epochs * pct / 100))
        target = min(target, total_epochs)
        targets.add(target)
    return targets


def _is_valid_webhook(url: str | None) -> bool:
    """Return True when the URL looks like a usable Slack webhook.

    Args:
        url (str | None): Candidate URL.

    Returns:
        bool: True when non-empty, not the placeholder, and HTTPS.
    """
    if not url:
        return False
    if url == PLACEHOLDER_URL:
        return False
    return url.startswith("http://") or url.startswith("https://")


def _resolve_webhook_url(raw: str | None) -> str | None:
    """Pick the effective webhook URL from config or environment.

    Args:
        raw (str | None): Value read from the YAML config.

    Returns:
        str | None: A usable URL, or ``None`` when no valid source exists.
    """
    if _is_valid_webhook(raw):
        return raw
    env_url = os.environ.get(ENV_WEBHOOK_URL)
    if _is_valid_webhook(env_url):
        return env_url
    if raw and raw != PLACEHOLDER_URL:
        LOGGER.warning(
            "Slack webhook URL is not a valid http(s) URL; notifier disabled"
        )
    return None


def _select_summary_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Pick a compact subset of validation metrics for message rendering.

    Prioritizes loss and AUPRC fields, then falls back to everything else
    truncated. Avoids flooding Slack with long metric dumps.

    Args:
        metrics (dict[str, float]): Full validation metrics.

    Returns:
        dict[str, float]: Up to 6 representative metrics.
    """
    priority = (
        "loss",
        "auc_pr_acceptor",
        "auc_pr_donor",
        "accuracy",
        "precision",
        "recall",
    )
    selected: dict[str, float] = {}
    for key in priority:
        if key in metrics:
            selected[key] = metrics[key]
    for key, value in metrics.items():
        if len(selected) >= 6:
            break
        if key not in selected:
            selected[key] = value
    return selected


def _format_scalar(value: Any) -> str:
    """Format a scalar metric value for Slack output.

    Args:
        value (Any): Metric value.

    Returns:
        str: Rendered representation.
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_fields(fields: dict[str, Any]) -> list[str]:
    """Render a ``key: value`` block as Slack mrkdwn bullets.

    Args:
        fields (dict[str, Any]): Fields to render.

    Returns:
        list[str]: Lines formatted as ``• *key*: value``.
    """
    return [f"• *{key}*: {value}" for key, value in fields.items()]
