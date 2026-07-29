"""Canonical English display labels for imputers and forecasters, so that all
paper figures use the same nomenclature as the LaTeX text/tables and the heatmap
(referee: uniformise Italian/code names to English)."""

IMP_LABELS = {
    'no_imp': 'No imputation',
    'media_glob': 'Mean (global)', 'media_cond': 'Mean (cond.)',
    'mediana_glob': 'Median (global)', 'mediana_cond': 'Median (cond.)',
    'forward_fill': 'Forward fill', 'seasonal_naive': 'Seasonal naive',
    'linear_interp': 'Linear interp.', 'lgb': 'LGB', 'dlinear': 'DLinear',
    'saits': 'SAITS', 'itransformer': 'iTransformer', 'timesnet': 'TimesNet',
    'imputeformer': 'ImputeFormer',
}

FC_LABELS = {
    'global_mean': 'Global Mean', 'dow_mean': 'DoW Mean', 'ma_k56': 'MA (K=56)',
    'croston': 'Croston', 'sba': 'SBA', 'tsb': 'TSB',
    'lgb_m5lags': 'LGB-M5', 'mlp_m5lags': 'MLP-M5', 'lgb_nolags': 'LGB',
    'mlp_nolags': 'MLP', 'tft': 'TFT', 'chronos_bolt': 'Chronos-bolt',
    'timesfm': 'TimesFM',
}


def imp(name):
    return IMP_LABELS.get(name, name)


def fc(name):
    return FC_LABELS.get(name, name)


def cell(imputer, forecaster):
    """Pretty 'Imputer / Forecaster' label."""
    return f'{imp(imputer)} / {fc(forecaster)}'
