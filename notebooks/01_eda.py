"""
01_eda.py — Analisi Esplorativa del Dataset FreshRetailNet-50K
=============================================================
PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting
di Prodotti Deperibili.

Questo script esegue il Passo 1 del piano sperimentale:
- Caricamento e ispezione dei file parquet (train + eval)
- Parsing dei campi orari (hours_sale, hours_stock_status)
- Statistiche descrittive e distribuzione stockout
- Visualizzazioni dei pattern temporali

Eseguire con: freshnet/bin/python notebooks/01_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

# ---------------------------------------------------------------------------
# 1. Caricamento dati
# ---------------------------------------------------------------------------
print('=' * 70)
print('1. CARICAMENTO DATI')
print('=' * 70)

df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))

print(f'Train: {df_train.shape[0]:,} righe × {df_train.shape[1]} colonne')
print(f'Eval:  {df_eval.shape[0]:,} righe × {df_eval.shape[1]} colonne')
print(f'\nColonne: {list(df_train.columns)}')
print(f'\nDtypes:\n{df_train.dtypes}')
print(f'\nMissing values (train): {df_train.isnull().sum().sum()} totali')
print(f'Missing values (eval):  {df_eval.isnull().sum().sum()} totali')

# ---------------------------------------------------------------------------
# 2. Ispezione campi orari
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('2. CAMPI ORARI: hours_sale e hours_stock_status')
print('=' * 70)

row0 = df_train.iloc[0]
print(f'hours_sale:         type={type(row0["hours_sale"]).__name__}, len={len(row0["hours_sale"])}')
print(f'hours_stock_status: type={type(row0["hours_stock_status"]).__name__}, len={len(row0["hours_stock_status"])}')
print(f'\nEsempio (riga 0):')
print(f'  hours_sale:         {row0["hours_sale"]}')
print(f'  hours_stock_status: {row0["hours_stock_status"]}')
print(f'  sale_amount:        {row0["sale_amount"]}')
print(f'  stock_hour6_22_cnt: {row0["stock_hour6_22_cnt"]}')

# Converti in matrici numpy per analisi vettorizzata
stock_arrays = np.array(df_train['hours_stock_status'].tolist())  # (4.5M, 24)
sale_arrays = np.array(df_train['hours_sale'].tolist())            # (4.5M, 24)

# Verifica coerenza hours_sale vs sale_amount
hourly_sum = sale_arrays.sum(axis=1)
match_pct = np.isclose(hourly_sum, df_train['sale_amount'].values, atol=0.01).mean()
print(f'\nCoerenza sum(hours_sale) ≈ sale_amount: {match_pct*100:.1f}%')

# Verifica stock_hour6_22_cnt
instock_hours_6_22 = (stock_arrays[:, 6:23] == 1).sum(axis=1)
match_stock = (df_train['stock_hour6_22_cnt'].values == instock_hours_6_22).mean()
print(f'Coerenza stock_hour6_22_cnt vs hours_stock_status[6:23]: {match_stock*100:.1f}%')

# ---------------------------------------------------------------------------
# 3. Cardinalità e struttura del dataset
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('3. CARDINALITA E STRUTTURA')
print('=' * 70)

for col in ['city_id', 'store_id', 'management_group_id',
            'first_category_id', 'second_category_id', 'third_category_id', 'product_id']:
    print(f'{col:25s}: {df_train[col].nunique():>5} valori unici')

print(f'\nPeriodo train: {df_train["dt"].min()} → {df_train["dt"].max()} ({df_train["dt"].nunique()} giorni)')
print(f'Periodo eval:  {df_eval["dt"].min()} → {df_eval["dt"].max()} ({df_eval["dt"].nunique()} giorni)')

n_series = df_train.groupby(['store_id', 'product_id']).ngroups
print(f'\nSerie uniche (store × product): {n_series:,}')
print(f'Righe per serie: {len(df_train) / n_series:.0f} (= 90 giorni)')

# ---------------------------------------------------------------------------
# 4. Statistiche descrittive
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('4. STATISTICHE DESCRITTIVE')
print('=' * 70)

print('\n--- sale_amount ---')
print(df_train['sale_amount'].describe().to_string())
print(f'\nZeri:    {(df_train["sale_amount"] == 0).sum():>10,} ({(df_train["sale_amount"] == 0).mean()*100:.1f}%)')
print(f'(0, 1]:  {((df_train["sale_amount"] > 0) & (df_train["sale_amount"] <= 1)).sum():>10,} ({((df_train["sale_amount"] > 0) & (df_train["sale_amount"] <= 1)).mean()*100:.1f}%)')
print(f'(1, 5]:  {((df_train["sale_amount"] > 1) & (df_train["sale_amount"] <= 5)).sum():>10,} ({((df_train["sale_amount"] > 1) & (df_train["sale_amount"] <= 5)).mean()*100:.1f}%)')
print(f'> 5:     {(df_train["sale_amount"] > 5).sum():>10,} ({(df_train["sale_amount"] > 5).mean()*100:.1f}%)')
print(f'P90={np.percentile(df_train["sale_amount"], 90):.1f}, P95={np.percentile(df_train["sale_amount"], 95):.1f}, P99={np.percentile(df_train["sale_amount"], 99):.1f}')

print('\n--- discount ---')
print(df_train['discount'].describe().to_string())

print('\n--- Meteo ---')
for col in ['precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']:
    s = df_train[col]
    print(f'{col:20s}: mean={s.mean():.2f}, std={s.std():.2f}, min={s.min():.2f}, max={s.max():.2f}')

# ---------------------------------------------------------------------------
# 5. Analisi stockout
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('5. ANALISI STOCKOUT')
print('=' * 70)

# CODIFICA: 0=in stock, 1=stockout (da documentazione ufficiale HuggingFace)
stockout_hourly_rate = (stock_arrays == 1).mean()
print(f'Tasso stockout globale (livello ora): {stockout_hourly_rate*100:.1f}%')

has_any_stockout = (stock_arrays.sum(axis=1) > 0)   # almeno un 1
full_stockout = (stock_arrays.sum(axis=1) == 24)     # tutti 1
no_stockout = (stock_arrays.sum(axis=1) == 0)        # tutti 0
print(f'Righe con almeno 1h di stockout: {has_any_stockout.sum():,} ({has_any_stockout.mean()*100:.1f}%)')
print(f'Righe con stockout completo:     {full_stockout.sum():,} ({full_stockout.mean()*100:.1f}%)')
print(f'Righe senza stockout (24h ok):   {no_stockout.sum():,} ({no_stockout.mean()*100:.1f}%)')

# Vendite durante stockout vs instock
print('\n--- Vendite per stato stock (livello orario) ---')
instock_mask = (stock_arrays == 0)
stockout_mask = (stock_arrays == 1)
print(f'Media vendita/ora durante IN STOCK: {sale_arrays[instock_mask].mean():.4f}')
print(f'Media vendita/ora durante STOCKOUT: {sale_arrays[stockout_mask].mean():.4f}')
print(f'Ore in-stock con vendite > 0: {(sale_arrays[instock_mask] > 0).mean()*100:.2f}%')
print(f'Ore stockout con vendite > 0: {(sale_arrays[stockout_mask] > 0).mean()*100:.2f}%')

# Stockout per ora del giorno
print('\n--- Tasso stockout per ora del giorno ---')
stockout_by_hour = (stock_arrays == 1).mean(axis=0)
for h in range(24):
    bar = '█' * int(stockout_by_hour[h] * 40)
    print(f'  {h:02d}:00  {stockout_by_hour[h]*100:5.1f}%  {bar}')

# Distribuzione stock_hour6_22_cnt
print('\n--- Distribuzione stock_hour6_22_cnt ---')
print(df_train['stock_hour6_22_cnt'].value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# 6. Stockout per dimensioni (categoria, città, giorno settimana)
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('6. STOCKOUT PER DIMENSIONI')
print('=' * 70)

df_train['stockout_rate'] = (stock_arrays == 1).mean(axis=1)
df_train['dt_parsed'] = pd.to_datetime(df_train['dt'])
df_train['day_of_week'] = df_train['dt_parsed'].dt.dayofweek

# Per categoria L1
print('\n--- Per Categoria L1 (top 10 per volume) ---')
cat_stats = df_train.groupby('first_category_id').agg(
    n_rows=('sale_amount', 'count'),
    mean_stockout=('stockout_rate', 'mean'),
    mean_sale=('sale_amount', 'mean'),
    n_products=('product_id', 'nunique')
).sort_values('n_rows', ascending=False).head(10)
print(cat_stats.to_string())

# Per città
print('\n--- Per Città ---')
city_stats = df_train.groupby('city_id').agg(
    n_rows=('sale_amount', 'count'),
    mean_stockout=('stockout_rate', 'mean'),
    mean_sale=('sale_amount', 'mean'),
    n_stores=('store_id', 'nunique')
).sort_values('n_rows', ascending=False)
print(city_stats.to_string())

# Per giorno della settimana
print('\n--- Per Giorno della Settimana ---')
day_names = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
dow_stats = df_train.groupby('day_of_week').agg(
    mean_stockout=('stockout_rate', 'mean'),
    mean_sale=('sale_amount', 'mean')
)
for dow, row in dow_stats.iterrows():
    print(f'  {day_names[dow]}: stockout={row["mean_stockout"]*100:.1f}%, vendite={row["mean_sale"]:.3f}')

# ---------------------------------------------------------------------------
# 7. Effetto discount, holiday, activity
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('7. EFFETTO VARIABILI ESOGENE')
print('=' * 70)

print('\n--- Discount ---')
df_train['discount_bin'] = pd.cut(df_train['discount'],
                                   bins=[0, 0.7, 0.85, 0.95, 1.0, 1.1],
                                   labels=['<0.7', '0.7-0.85', '0.85-0.95', '0.95-1.0', '>1.0'])
disc_stats = df_train.groupby('discount_bin', observed=True).agg(
    n_rows=('sale_amount', 'count'),
    mean_sale=('sale_amount', 'mean'),
    mean_stockout=('stockout_rate', 'mean')
)
print(disc_stats.to_string())

print('\n--- Holiday Flag ---')
hol_stats = df_train.groupby('holiday_flag').agg(
    n_rows=('sale_amount', 'count'),
    mean_sale=('sale_amount', 'mean'),
    mean_stockout=('stockout_rate', 'mean')
)
print(hol_stats.to_string())

print('\n--- Activity Flag ---')
act_stats = df_train.groupby('activity_flag').agg(
    n_rows=('sale_amount', 'count'),
    mean_sale=('sale_amount', 'mean'),
    mean_stockout=('stockout_rate', 'mean')
)
print(act_stats.to_string())

# ---------------------------------------------------------------------------
# 8. Confronto Train vs Eval
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('8. CONFRONTO TRAIN vs EVAL')
print('=' * 70)

stock_eval = np.array(df_eval['hours_stock_status'].tolist())
print(f'Eval: {df_eval["dt"].min()} → {df_eval["dt"].max()} ({df_eval["dt"].nunique()} giorni)')
print(f'Serie eval: {df_eval.groupby(["store_id", "product_id"]).ngroups:,}')
print(f'Stockout rate eval (orario): {(stock_eval == 1).mean()*100:.1f}%')

for col in ['sale_amount', 'discount', 'stock_hour6_22_cnt', 'precpt', 'avg_temperature', 'avg_humidity']:
    print(f'{col:25s}: train={df_train[col].mean():.3f}, eval={df_eval[col].mean():.3f}')

print('\n⚠ NOTA: eval ha temperatura più alta (+7°C) e più pioggia.')
print('  Covariate shift da tenere in conto per la valutazione.')

# ---------------------------------------------------------------------------
# 9. Visualizzazioni
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('9. GENERAZIONE VISUALIZZAZIONI')
print('=' * 70)

# Fig 1: Stockout e vendite per ora
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
stockout_by_hour = (stock_arrays == 1).mean(axis=0) * 100  # 1=stockout
sales_by_hour = sale_arrays.mean(axis=0)

ax1.bar(range(24), stockout_by_hour, color='#e74c3c', alpha=0.7)
ax1.set_ylabel('Stockout Rate (%)')
ax1.set_title('Tasso di Stockout per Ora del Giorno (stock_status=1)')
ax1.axhline(y=24.9, color='gray', linestyle='--', alpha=0.5, label='Media globale (24.9%)')
ax1.legend()

ax2.bar(range(24), sales_by_hour, color='#3498db', alpha=0.7)
ax2.set_ylabel('Vendite Medie / Ora')
ax2.set_xlabel('Ora del Giorno')
ax2.set_title('Vendite Medie per Ora del Giorno')
ax2.set_xticks(range(24))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '01_hourly_stockout_sales.png'), bbox_inches='tight')
plt.close()
print('  01_hourly_stockout_sales.png')

# Fig 2: Distribuzione sale_amount
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sale_pos = df_train['sale_amount'][df_train['sale_amount'] > 0]
ax1.hist(sale_pos, bins=100, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.3)
ax1.set_xlabel('sale_amount (> 0)')
ax1.set_ylabel('Frequenza')
ax1.set_title(f'Distribuzione sale_amount (escl. zeri, n={len(sale_pos):,})')
ax1.set_yscale('log')

df_train['stockout_pct'] = df_train['stockout_rate'] * 100
df_train['stockout_bin'] = pd.cut(df_train['stockout_pct'], bins=[0, 25, 50, 75, 100],
                                   labels=['0-25%', '25-50%', '50-75%', '75-100%'], include_lowest=True)
sample = df_train.sample(50000, random_state=42)
sns.boxplot(data=sample, x='stockout_bin', y='sale_amount', ax=ax2, showfliers=False)
ax2.set_xlabel('Stockout Rate (% ore giorno)')
ax2.set_ylabel('sale_amount')
ax2.set_title('Vendite per Livello di Stockout')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '02_sale_amount_distribution.png'), bbox_inches='tight')
plt.close()
print('  02_sale_amount_distribution.png')

# Fig 3: Andamento temporale
daily = df_train.groupby('dt_parsed').agg(
    mean_sale=('sale_amount', 'mean'),
    mean_stockout=('stockout_rate', 'mean')
)
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2_twin = ax1.twinx()
ax1.plot(daily.index, daily['mean_sale'], color='#3498db', linewidth=1.5, label='Vendite medie')
ax2_twin.plot(daily.index, daily['mean_stockout'] * 100, color='#e74c3c', linewidth=1.5, alpha=0.6, label='Stockout %')
ax1.set_xlabel('Data')
ax1.set_ylabel('Vendite Medie', color='#3498db')
ax2_twin.set_ylabel('Stockout Rate (%)', color='#e74c3c')
ax1.set_title('Andamento Temporale: Vendite e Stockout')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '03_temporal_trend.png'), bbox_inches='tight')
plt.close()
print('  03_temporal_trend.png')

# Fig 4: Heatmap vendite (ora × giorno settimana)
hourly_sales_matrix = np.zeros((7, 24))
for dow in range(7):
    mask = df_train['day_of_week'] == dow
    hourly_sales_matrix[dow] = sale_arrays[mask].mean(axis=0)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(hourly_sales_matrix, ax=ax, cmap='YlOrRd',
            xticklabels=range(24), yticklabels=day_names,
            cbar_kws={'label': 'Vendite medie/ora'})
ax.set_xlabel('Ora del Giorno')
ax.set_ylabel('Giorno della Settimana')
ax.set_title('Heatmap Vendite: Ora × Giorno Settimana')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '04_heatmap_sales_dow_hour.png'), bbox_inches='tight')
plt.close()
print('  04_heatmap_sales_dow_hour.png')

# Fig 5: Heatmap stockout (ora × giorno settimana)
hourly_stockout_matrix = np.zeros((7, 24))
for dow in range(7):
    mask = df_train['day_of_week'] == dow
    hourly_stockout_matrix[dow] = (stock_arrays[mask] == 1).mean(axis=0) * 100  # 1=stockout

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(hourly_stockout_matrix, ax=ax, cmap='Reds',
            xticklabels=range(24), yticklabels=day_names,
            cbar_kws={'label': 'Stockout Rate (%)'}, vmin=0, vmax=50)
ax.set_xlabel('Ora del Giorno')
ax.set_ylabel('Giorno della Settimana')
ax.set_title('Heatmap Stockout (stock_status=1): Ora × Giorno Settimana')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '05_heatmap_stockout_dow_hour.png'), bbox_inches='tight')
plt.close()
print('  05_heatmap_stockout_dow_hour.png')

# Fig 6: Vendite vs meteo
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(df_train), 20000, replace=False)
for ax, col, label in zip(axes, ['avg_temperature', 'avg_humidity', 'precpt'],
                           ['Temperatura (°C)', 'Umidità (%)', 'Precipitazioni']):
    ax.scatter(df_train[col].iloc[sample_idx], df_train['sale_amount'].iloc[sample_idx],
               alpha=0.05, s=3, color='#3498db')
    ax.set_xlabel(label)
    ax.set_ylabel('sale_amount')
axes[0].set_title('Vendite vs Temperatura')
axes[1].set_title('Vendite vs Umidità')
axes[2].set_title('Vendite vs Precipitazioni')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '06_sales_vs_weather.png'), bbox_inches='tight')
plt.close()
print('  06_sales_vs_weather.png')

# Fig 7: Stockout e vendite per categoria
cat_stats_plot = df_train.groupby('first_category_id').agg(
    mean_stockout=('stockout_rate', 'mean'),
    mean_sale=('sale_amount', 'mean'),
    count=('sale_amount', 'count')
).sort_values('count', ascending=False).head(10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cats = [str(c) for c in cat_stats_plot.index]
ax1.barh(cats, cat_stats_plot['mean_stockout'] * 100, color='#e74c3c', alpha=0.7)
ax1.set_xlabel('Stockout Rate (%)')
ax1.set_title('Stockout Rate per Categoria L1 (Top 10)')
ax1.invert_yaxis()
ax2.barh(cats, cat_stats_plot['mean_sale'], color='#3498db', alpha=0.7)
ax2.set_xlabel('Vendite Medie')
ax2.set_title('Vendite Medie per Categoria L1 (Top 10)')
ax2.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '07_category_stockout_sales.png'), bbox_inches='tight')
plt.close()
print('  07_category_stockout_sales.png')

# ---------------------------------------------------------------------------
# 10. Riepilogo e implicazioni per il modello PINN
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('10. RIEPILOGO E IMPLICAZIONI PER PINN-RETAIL')
print('=' * 70)

print("""
DATASET:
  - 4.5M righe train (90 giorni) + 350K eval (7 giorni)
  - 50,000 serie (898 negozi x 865 prodotti)
  - 19 colonne, zero valori mancanti
  - Campi orari: array numpy di 24 elementi (ore 0-23)
  - Codifica: 0=in stock, 1=stockout (da documentazione HuggingFace)

STOCKOUT (il problema centrale):
  - 24.9% delle ore-slot sono in stockout (stock_status=1)
  - 60.0% delle righe-giorno hanno almeno 1h di stockout
  - 3.8% hanno stockout completo (24h)
  - 40.0% non hanno alcun stockout
  - Media 6 ore di stockout per giorno

VENDITE vs STOCK STATUS:
  - Media vendite/ora IN STOCK:  0.054 (28.6% ore con vendite > 0)
  - Media vendite/ora STOCKOUT:  0.004 (2.9% ore con vendite > 0)
  - Rapporto: 12.9x — il censoring funziona come atteso

PATTERN TEMPORALI:
  - Stockout minimo al mattino (~5% ore 7-8, dopo rifornimento)
  - Stockout cresce monotonamente fino a sera (~42% ore 22-23)
  - Vendite: due picchi (mattina 8-10, pomeriggio 15-17)
  - Weekend: vendite piu alte (+25%), stockout simile
  - Trend crescente nel tempo (vendite in aumento)

VARIABILI ESOGENE:
  - Discount < 0.7 -> vendite +59% vs prezzo pieno
  - Holiday: vendite +27%
  - Activity: effetto modesto
  - Meteo: relazione debole con vendite aggregate

CONFRONTO TRAIN vs EVAL:
  - Eval e immediatamente successivo (no gap)
  - Temperatura eval +7C (29C vs 22C)
  - Precipitazioni eval +59%
  -> Covariate shift stagionale da monitorare

IMPLICAZIONI PER IL MODELLO:
  1. Il 24.9% di ore censurate e significativo — il 75% del segnale e pulito
  2. Il pattern monotono di stockout intra-giornaliero supporta il vincolo L_cons
  3. Prodotti ad alta domanda hanno piu stockout -> dove il PINN aggiunge valore
  4. Il modello deve gestire la stagionalita intra-giornaliera (bimodale)
  5. Le 50K serie con 90 punti/serie permettono un modello globale robusto
""")

print('EDA completata. Figure salvate in notebooks/figures/')
