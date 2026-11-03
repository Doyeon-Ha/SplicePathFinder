"""Hyena inference engine for Evo2 (StripedHyena2).

Three operator families are exercised by Evo2 blocks:

  * parallel_fir(gate=False)  -- outer FIR used by all hyena blocks before any
                                 channel split (input projection convolution).
  * parallel_fir(gate=True)   -- inner FIR cascade used by hcm/hcs blocks
                                 (x1 * v gated, then convolved by `h`,
                                 then multiplied by x2 postgate).
  * parallel_iir              -- modal-form IIR (long convolution via FFT)
                                 used by hcl blocks; poles + residues parameterize
                                 a stable, long-range linear filter.

Sequential step paths (step_fir / step_iir) are used during generation.

Layout conventions match vortex exactly so checkpoints are bit-identical.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


IIR_PREFILL_MODES = ["recurrence", "modal-fft"]


def adjust_filter_shape_for_broadcast(u, h):
    h = h.squeeze()
    if len(u.shape) > len(h.shape):
        h = h.unsqueeze(0)
    if len(u.shape) > 3:
        h = h.unsqueeze(1)
    return h


def fftconv_func(u, k, D, dropout_mask, gelu=True, k_rev=None, bidirectional=False, **kwargs):
    """FFT convolution for long FIR filters (length >= 128 path)."""
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    k_f = adjust_filter_shape_for_broadcast(u, k_f)
    k = k.squeeze()

    if bidirectional:
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
        k, k2 = k.split(k.shape[1] // 2, dim=1)
        k2_f = torch.fft.rfft(k2, n=fft_size) / fft_size
        y1 = u_f * k_f
        y2 = u_f.conj() * k2_f.conj()
        y = torch.fft.irfft(y1 + y2, n=fft_size, norm="forward")[..., :seqlen]
    else:
        if k_rev is not None:
            k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
            k_f = k_f + k_rev_f.conj()
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


def _column_split(x, num_heads, head_size):
    """Compatibility helper for column_split_hyena=True (not used by Evo2)."""
    x = x.reshape(x.shape[0], num_heads, 3 * head_size, x.shape[2])
    x2 = x[:, :, :head_size].reshape(x.shape[0], -1, x.shape[-1])
    x1 = x[:, :, head_size : 2 * head_size].reshape(x.shape[0], -1, x.shape[-1])
    v = x[:, :, 2 * head_size :].reshape(x.shape[0], -1, x.shape[-1])
    return x2, x1, v


class HyenaInferenceEngine:
    def __init__(
        self,
        layer_idx: int | None = None,
        iir_prefill_style: str = "modal-fft",
        hyena_flip_x1x2: bool = False,
    ) -> None:
        assert iir_prefill_style in IIR_PREFILL_MODES, iir_prefill_style
        self.iir_prefill_style = iir_prefill_style
        self.layer_idx = layer_idx
        self.low_mem_mode = False
        self.hyena_flip_x1x2 = hyena_flip_x1x2

    # ---------------------------------------------------------------- FIR
    def parallel_fir(
        self,
        fir_fn,
        u,
        weight,
        bias,
        L,
        dims,
        groups=None,
        gated_bias=False,
        column_split_hyena=False,
        dim_last=True,
        fir_length=3,
        gate=False,
        inference_params=None,
        padding_mask=None,
    ):
        L = u.shape[1] if dim_last else u.shape[2]
        if gate:
            hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
            if column_split_hyena:
                x2, x1, v = _column_split(u, num_attention_heads, hidden_size_per_attention_head)
            else:
                x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
            if self.hyena_flip_x1x2:
                x1, x2 = x2, x1
            u = x1 * v

        if fir_length >= 128:
            with torch.autocast("cuda"):
                z = fftconv_func(
                    u.to(torch.float32),
                    weight[:, :, :L].to(torch.float32),
                    bias,
                    None,
                    gelu=False,
                    bidirectional=False,
                    groups=groups,
                )
                z = z.to(u.dtype)
        else:
            if dim_last:
                u = u.permute(0, 2, 1)  # B, D, L

            z = fir_fn(
                u.to(torch.float32),
                weight.to(torch.float32),
                bias=None,
                stride=1,
                padding=fir_length - 1,
                groups=u.shape[1],
            )[..., :L]

            z = z.to(u.dtype)

            if bias is not None:
                if gated_bias:
                    z = z + bias[None, :, None] * u
                else:
                    z = z + bias[None, :, None]

        if isinstance(padding_mask, torch.Tensor):
            z = z * padding_mask[:, None]

        if gate:
            z = x2 * z

        if inference_params is not None:
            fir_state = u[..., -fir_length + 1 :]
        else:
            fir_state = None

        return z, fir_state

    # ---------------------------------------------------------------- IIR
    def parallel_iir(
        self,
        z_pre,
        h,
        D,
        L,
        poles,
        residues,
        t,
        dims,
        layer_idx,
        inference_params=None,
        prefill_style: str = "fft",
        fftconv_fn=None,
        padding_mask=None,
        use_flashfft: bool = False,
        column_split_hyena: bool = False,
        long_fir_threshold: int | None = None,
    ):
        fft_size = 2 * L
        hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
        if column_split_hyena:
            x2, x1, v = _column_split(z_pre, num_attention_heads, hidden_size_per_attention_head)
        else:
            x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        x1v = x1 * v

        X_s = None
        if inference_params is not None and prefill_style == "recurrence":
            y = self.prefill_via_direct_recurrence(
                inference_params=inference_params, x1v=x1v, L=L,
                poles=poles, residues=residues,
            )
        else:
            if use_flashfft and (L % 2) == 0:
                y = fftconv_fn(
                    x1v.to(dtype=torch.bfloat16).contiguous(),
                    h.to(dtype=torch.float32),
                )
            elif long_fir_threshold is None:
                H = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
                X_s = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)
                X = X_s[..., : H.shape[-1]]
                if len(z_pre.shape) > 3:
                    H = H.unsqueeze(1)
                y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]
            else:
                assert h.shape[0] == 1, "batch size must be 1 for long_fir_threshold"
                h = h[0][:, None]
                h = h[..., :long_fir_threshold]
                y = F.conv1d(
                    x1v, h.to(dtype=x1v.dtype),
                    stride=1, groups=x1v.shape[1],
                    padding=h.shape[-1] - 1,
                )[..., :L]

        y = y.to(dtype=x1v.dtype)
        y = (y + x1v * D.unsqueeze(-1)) * x2

        if inference_params is not None and prefill_style == "fft":
            self.prefill_via_modal_fft(
                inference_params=inference_params, x1v=x1v, X_s=X_s, L=L,
                t=t, poles=poles, dims=dims, layer_idx=layer_idx,
                use_flashfft=use_flashfft, fftconv_fn=fftconv_fn,
            )

        return y.permute(0, 2, 1)

    # --------------------------------------------------------- step (decode)
    def step_fir(self, u, fir_state, weight, bias=None, gated_bias=False, flip_filter=False):
        """Single-step FIR. fir_state holds the last (filter_len - 1) inputs."""
        weight = weight.squeeze()
        cache_size = fir_state.shape[-1]
        filter_length = weight.shape[-1]
        if flip_filter:
            weight = weight.flip(-1)
            weight = weight[..., -cache_size - 1 :].unsqueeze(0)
        else:
            weight = weight[..., : cache_size + 1].unsqueeze(0)

        input_dtype = u.dtype
        weight = weight.to(torch.float32)
        u = u.to(torch.float32)
        fir_state = fir_state.to(torch.float32)
        bias = bias.to(torch.float32) if bias is not None else None

        h0, h = weight[..., -1], weight[..., :-1]
        y = h0 * u + torch.sum(fir_state * h, dim=-1)

        if bias is not None:
            if gated_bias:
                y = y + bias * u
            else:
                y = y + bias

        if cache_size < filter_length - 1:
            fir_state = torch.cat([fir_state, u[..., None]], dim=-1)
        else:
            fir_state = torch.roll(fir_state, -1, dims=2)
            fir_state[..., -1] = u

        return y.to(input_dtype), fir_state

    def step_iir(self, x2, x1, v, D, residues, poles, iir_state, iir_groups=1):
        x1v = x1 * v
        # `poles` arg contains log_poles (real, in modal form for evo2)
        poles = torch.exp(poles)
        poles = poles[..., 0][None]
        residues = residues[None]
        iir_state = poles * iir_state + x1v[..., None]
        res_state = torch.sum(residues * iir_state, dim=-1)

        if iir_groups > 1:
            raise NotImplementedError
        y = x2 * (res_state + D * x1v)
        return y, iir_state

    def prefill_via_direct_recurrence(self, inference_params, x1v, L, residues, poles, *args, **kwargs):
        state_dim = poles.shape[1]
        x1v_ = x1v[..., None, None]
        x1v_ = x1v_.repeat(1, 1, 1, state_dim, 2)
        x1v_[..., 1] = 0

        state = 0 * x1v_[:, :, 0]
        output = 0 * x1v_[:, :, :, 0, 0]

        poles = poles[:, :, 0][None]
        residues = residues[:, :, 0][None].repeat(x1v_.shape[0], 1, 1, 1)

        for i in range(L):
            state[..., 0] = poles[..., 0] * state[..., 0] - poles[..., 1] * state[..., 1] + x1v_[:, :, i, :, 0]
            state[..., 1] = poles[..., 0] * state[..., 1] + poles[..., 1] * state[..., 0] + x1v_[:, :, i, :, 1]
            output[:, :, i] = torch.sum(residues * state, dim=-2)[..., 0]

        inference_params.state_dict[self.layer_idx] = state.to(dtype=torch.float32)
        return output

    def prefill_via_modal_fft(
        self, inference_params, x1v, L, poles, t, dims, layer_idx,
        X_s=None, use_flashfft=False, fftconv_fn=None,
        state_dtype=torch.float32, *args, **kwargs,
    ):
        """Compute IIR state via a single FFT.

        Evo2 uses *real* `log_poles` (not the complex view-as-real layout
        that Evo1 uses), so the impulse-response IFFT is mathematically real;
        the imaginary component is FFT round-off. We take ``.real`` explicitly
        instead of relying on torch's lossy complex->real cast, which avoids
        the "Casting complex values to real discards the imaginary part"
        UserWarning at every decoded token.
        """
        hidden_size, _, _, state_size, hyena_filter_groups = dims

        assert X_s is not None
        bs = x1v.shape[0]
        fft_size = 2 * L
        state_s = (poles.to(torch.float32) * t).exp()
        state_S = torch.fft.fft(state_s, n=fft_size).repeat(bs, 1, 1, 1)
        if hyena_filter_groups > 1:
            state_S = state_S.repeat_interleave(hidden_size // hyena_filter_groups, 1)
        state = torch.fft.ifft(X_s[..., None, :] * state_S, n=fft_size)
        inference_params.state_dict[layer_idx] = state[..., L - 1].real.to(dtype=state_dtype)
