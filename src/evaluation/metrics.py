"""Reusable evaluation metrics for demand forecasting models.

All functions are pure (no side effects) and operate on numpy arrays.
Designed to be called from any model's evaluation notebook.
"""

import numpy as np
import pandas as pd


def compute_metrics(y_pred, y_obs, stock_status):
    """Compute WAPE and WPE on 3 subsets: overall, in-stock, stockout.

    Args:
        y_pred: np.array — predicted values (flattened or any shape)
        y_obs: np.array — observed sales (same shape as y_pred)
        stock_status: np.array — 0=in-stock, 1=stockout (same shape)

    Returns:
        dict with keys:
            wape_overall, wape_instock, wape_stockout,
            wpe_overall, wpe_instock, wpe_stockout,
            n_overall, n_instock, n_stockout
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_obs = np.asarray(y_obs, dtype=np.float64).ravel()
    stock_status = np.asarray(stock_status).ravel()

    assert y_pred.shape == y_obs.shape == stock_status.shape, (
        f"Shape mismatch: y_pred={y_pred.shape}, y_obs={y_obs.shape}, "
        f"stock_status={stock_status.shape}"
    )

    instock_mask = stock_status == 0
    stockout_mask = stock_status == 1

    results = {}

    for suffix, mask in [
        ("overall", np.ones(len(y_pred), dtype=bool)),
        ("instock", instock_mask),
        ("stockout", stockout_mask),
    ]:
        yp = y_pred[mask]
        yo = y_obs[mask]
        n = int(mask.sum())

        results[f"n_{suffix}"] = n

        sum_abs_obs = np.abs(yo).sum()
        sum_obs = yo.sum()

        if n == 0 or sum_abs_obs == 0:
            results[f"wape_{suffix}"] = np.nan
        else:
            results[f"wape_{suffix}"] = np.abs(yp - yo).sum() / sum_abs_obs

        if n == 0 or sum_obs == 0:
            results[f"wpe_{suffix}"] = np.nan
        else:
            results[f"wpe_{suffix}"] = (yp - yo).sum() / sum_obs

    return results


def compute_metrics_per_series(df, y_pred_col, y_obs_col, stock_col,
                               group_cols=('store_id', 'product_id')):
    """Compute WAPE/WPE per-series, aggregating all days within each series.

    For each series, all days in the split are aggregated:
    - WAPE = Σ|y_pred - y_obs| / Σ|y_obs|
    - WPE  = Σ(y_pred - y_obs) / Σ y_obs
    On 3 subsets: overall, instock, stockout.

    Args:
        df: DataFrame with columns group_cols + [y_pred_col, y_obs_col, stock_col].
            The y_pred_col, y_obs_col, stock_col columns contain lists/arrays of 24 hourly values.
        y_pred_col: column name for predicted values (list of 24)
        y_obs_col: column name for observed values (list of 24)
        stock_col: column name for stock status (list of 24, 0=in-stock, 1=stockout)
        group_cols: columns defining a series

    Returns:
        pd.DataFrame with one row per series and columns:
        *group_cols, wape_overall, wape_instock, wape_stockout,
        wpe_overall, wpe_instock, wpe_stockout,
        n_overall, n_instock, n_stockout
    """
    group_cols = list(group_cols)
    records = []

    for keys, grp in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        y_pred = np.array(grp[y_pred_col].tolist())
        y_obs = np.array(grp[y_obs_col].tolist())
        stock = np.array(grp[stock_col].tolist())

        m = compute_metrics(y_pred, y_obs, stock)
        row = dict(zip(group_cols, keys))
        row.update(m)
        records.append(row)

    return pd.DataFrame(records)


def format_metrics_table(results_dict, model_name="Model"):
    """Format a dict of {split_name: metrics_dict} into a printable table.

    Args:
        results_dict: dict like {"train": metrics, "val": metrics, "test": metrics}
        model_name: name to display in the header

    Returns:
        str — formatted table
    """
    header = f"\n{'=' * 72}\n  {model_name} — Evaluation Results\n{'=' * 72}\n"

    col_headers = f"{'Split':<8} {'WAPE_all':>10} {'WAPE_in':>10} {'WAPE_so':>10} {'WPE_all':>10} {'N_hours':>12}\n"
    col_headers += "-" * 72 + "\n"

    rows = ""
    for split_name, m in results_dict.items():
        def fmt(v):
            return f"{v:>10.4f}" if not np.isnan(v) else f"{'N/A':>10}"

        rows += (
            f"{split_name:<8} "
            f"{fmt(m['wape_overall'])} "
            f"{fmt(m['wape_instock'])} "
            f"{fmt(m['wape_stockout'])} "
            f"{fmt(m['wpe_overall'])} "
            f"{m['n_overall']:>12,}\n"
        )

    return header + col_headers + rows
