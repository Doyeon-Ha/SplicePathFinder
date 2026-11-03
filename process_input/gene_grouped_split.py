"""SpliceVarDB 변이의 gene-grouped train/val/test 분할.

같은 ``gene_symbol``의 변이가 둘 이상의 분할에 들어가지 않도록 그룹 단위로
나눈다(데이터 누수 방지). 같은 seed면 동일 멤버십이 재현된다.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

_MISSING_GENE_TOKENS: frozenset[str] = frozenset({"", "nan", "none", "na", "."})


def _decode_gene(value: object) -> str:
    """gene_symbol 원소를 소문자 strip 문자열로 디코드한다.

    Args:
        value (object): bytes 또는 str.

    Returns:
        str: 정규화된 gene symbol(빈 값은 빈 문자열).
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    return str(value).strip()


def gene_grouped_split(
    gene_symbols: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gene 단위로 train/val/test 인덱스를 나눈다.

    Args:
        gene_symbols (np.ndarray): shape ``(N,)`` 변이별 gene symbol(str/bytes).
        labels (np.ndarray): shape ``(N,)`` 이진 라벨(분할별 비율 로깅용).
        seed (int): 결정적 분할을 위한 seed.
        ratios (tuple[float, float, float]): train/val/test 비율(합=1).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 원본 배열 기준
            (train_idx, val_idx, test_idx). 각각 오름차순 정렬.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios 합은 1이어야 한다: {ratios}")
    genes = np.array([_decode_gene(g) for g in gene_symbols])
    is_valid = np.array([g.lower() not in _MISSING_GENE_TOKENS for g in genes])
    n_missing = int((~is_valid).sum())
    if n_missing:
        logger.warning("gene_symbol 결측 변이 %d개 제외", n_missing)
    idx = np.nonzero(is_valid)[0]
    groups = genes[idx]

    test_size = ratios[2]
    splitter_test = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    rest_local, test_local = next(splitter_test.split(idx, groups=groups))

    val_fraction = ratios[1] / (ratios[0] + ratios[1])
    splitter_val = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_local2, val_local2 = next(
        splitter_val.split(idx[rest_local], groups=groups[rest_local])
    )

    train_idx = np.sort(idx[rest_local][train_local2])
    val_idx = np.sort(idx[rest_local][val_local2])
    test_idx = np.sort(idx[test_local])

    _log_split("train", train_idx, labels, genes)
    _log_split("val", val_idx, labels, genes)
    _log_split("test", test_idx, labels, genes)
    return train_idx, val_idx, test_idx


def _log_split(
    name: str,
    indices: np.ndarray,
    labels: np.ndarray,
    genes: np.ndarray,
) -> None:
    """분할 크기·양성 비율·gene 수를 로깅한다.

    Args:
        name (str): 분할 이름.
        indices (np.ndarray): 해당 분할 인덱스.
        labels (np.ndarray): 전체 라벨 배열.
        genes (np.ndarray): 전체 gene 배열.
    """
    n = len(indices)
    pos = float(labels[indices].mean()) if n else 0.0
    n_genes = len(np.unique(genes[indices])) if n else 0
    logger.info("%s: n=%d positive_ratio=%.3f genes=%d", name, n, pos, n_genes)


def assert_no_gene_leakage(
    gene_symbols: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """분할 간 gene 교집합이 공집합인지 검증한다.

    Args:
        gene_symbols (np.ndarray): shape ``(N,)`` gene symbol.
        train_idx (np.ndarray): train 인덱스.
        val_idx (np.ndarray): val 인덱스.
        test_idx (np.ndarray): test 인덱스.

    Raises:
        ValueError: 두 분할이 같은 gene을 공유하면 발생.
    """
    genes = np.array([_decode_gene(g) for g in gene_symbols])
    g_train = set(genes[train_idx])
    g_val = set(genes[val_idx])
    g_test = set(genes[test_idx])
    for a_name, a_set, b_name, b_set in (
        ("train", g_train, "val", g_val),
        ("train", g_train, "test", g_test),
        ("val", g_val, "test", g_test),
    ):
        overlap = a_set & b_set
        if overlap:
            raise ValueError(f"gene 누수: {a_name}∩{b_name}에 {len(overlap)}개 공유")
