"""
Stored training-only climatology tables ("encodings") shared by the training experiment and the
live forecaster, so live inputs are computed exactly as validation/test inputs were.

A table holds, per group (grid cell, or one pooled group), the TRAIN-split sum and count of a value
for each key level, e.g. [['target_month', 'target_hour'], ['target_hour'], []]. Lookup uses the
first level with at least `min_count` training rows ([] needs any training row) - identical to
train_and_validate_hmm.train_encoding for non-training rows (training rows there use leave-one-out).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

POOLED = '__all__'


def _k(v: Any) -> Any:
    return v.item() if hasattr(v, 'item') else v


def fit_table(df: pd.DataFrame, values: np.ndarray, levels: List[List[str]], per_cell: bool = True,
              min_count: int = 20) -> Dict[str, Any]:
    """Build a table from the TRAIN rows of df (columns 'split', the key columns and 'cell' if per_cell)."""
    v = np.asarray(values, dtype=float)
    train = (df['split'] == 'train').to_numpy() & np.isfinite(v)
    stats: Dict[str, List[Dict]] = {}
    groups = (df['cell'].to_numpy() if per_cell else np.full(len(df), POOLED, dtype=object))
    for g in pd.unique(groups[train]):
        m = train & (groups == g)
        sub = df.loc[m]
        per_level = []
        for keys in levels:
            if keys:
                agg = pd.Series(v[m]).groupby([sub[k].to_numpy() for k in keys]).agg(['sum', 'count'])
                per_level.append({(tuple(_k(x) for x in idx) if isinstance(idx, tuple) else (_k(idx),)):
                                  (float(s), int(c)) for idx, s, c in zip(agg.index, agg['sum'], agg['count'])})
            else:
                per_level.append({(): (float(v[m].sum()), int(m.sum()))})
        stats[str(g)] = per_level
    return {'levels': [list(k) for k in levels], 'per_cell': per_cell, 'min_count': int(min_count), 'stats': stats}


def lookup(table: Dict[str, Any], row: Dict[str, Any], cell: Optional[str] = None) -> float:
    """Training mean for one row (dict of key values), with the table's fallback rule; NaN if none."""
    group = str(cell) if table['per_cell'] else POOLED
    per_level = table['stats'].get(group)
    if per_level is None:
        return float('nan')
    for keys, entries in zip(table['levels'], per_level):
        s_c = entries.get(tuple(_k(row[k]) for k in keys))
        if s_c is None:
            continue
        s, c = s_c
        if (c >= max(1, table['min_count'])) if keys else (c > 0):
            return s / c
    return float('nan')


def logit_clip(p) -> np.ndarray:
    """log(p/(1-p)) with p clipped to [0.005, 0.995] (same as the experiment's _logit_clip)."""
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return np.log(p / (1 - p))


def apply_table(table: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Vectorised lookup for every row of df (uses df['cell'] when the table is per cell)."""
    cols = sorted({k for keys in table['levels'] for k in keys})
    cells = df['cell'].to_numpy() if table['per_cell'] else [None] * len(df)
    recs = df[cols].to_dict('records') if cols else [{}] * len(df)
    return np.array([lookup(table, r, c) for r, c in zip(recs, cells)], dtype=float)
