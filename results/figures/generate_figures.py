
"""
UrbanPulse — Research Figure Generator (Step 13B Part 2)
=========================================================
Generates all publication/presentation-quality PNG figures from FROZEN
test evaluation artifacts. No model retraining or data fabrication.

Outputs (results/figures/):
  1.  aqi_mae_hmm_vs_mrf.png
  2.  aqi_rmse_hmm_vs_mrf.png
  3.  rainfall_mae_hmm_vs_mrf.png
  4.  rainfall_rmse_hmm_vs_mrf.png
  5.  aqi_zonewise_mae.png
  6.  aqi_horizon_performance.png
  7.  alert_distribution.png
  8.  aqi_confusion_matrix.png         [valid: 978 actual AQI > 200 events]
  9.  rainfall_confusion_matrix.png    [valid: 1553 actual 24h >= 64.5mm events]
  10. temperature_evaluation_unavailable.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ── Setup ─────────────────────────────────────────────────────────────────────
ROOT = 'c:/Users/Tanvi/Desktop/Books/Sem-5/PR/UrbanPulse'
os.chdir(ROOT)
FIGURES_DIR = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'axes.titlecolor': '#e6edf3',
    'axes.grid': True,
    'grid.color': '#21262d',
    'grid.linewidth': 0.6,
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'legend.facecolor': '#1c2128',
    'legend.edgecolor': '#30363d',
    'legend.labelcolor': '#c9d1d9',
    'legend.fontsize': 9,
    'font.family': 'DejaVu Sans',
    'savefig.facecolor': '#0d1117',
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

BLUE_HMM   = '#6366f1'   # indigo — HMM only
TEAL_MRF   = '#06b6d4'   # cyan   — HMM+MRF
RED_WARN   = '#ef4444'
AMBER_WATCH = '#f59e0b'
GREEN_NORM = '#22c55e'
GRAY_UNMON = '#6b7280'
ACCENT     = '#a78bfa'

def savefig(fname, fig):
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close(fig)

def subtitle_text(ax, txt):
    ax.text(0.5, -0.14, txt, transform=ax.transAxes,
            ha='center', va='top', fontsize=7.5, color='#8b949e', style='italic')

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading frozen test artifacts...")
df_metrics = pd.read_csv('results/metrics/test_metrics_comparison.csv')
df_alerts  = pd.read_csv('results/alerts/test_alerts.csv', low_memory=False)
df_preds   = pd.read_csv('results/predictions/test_predictions.csv', low_memory=False)
df_test    = pd.read_csv('data/processed/test.csv', low_memory=False)
print(f"  Metrics: {len(df_metrics)} rows")
print(f"  Alerts: {len(df_alerts):,} rows")
print(f"  Predictions: {len(df_preds):,} rows")
print(f"  Test observations: {len(df_test):,} rows")

HORIZONS = ['1h', '3h', '6h', '24h']
HORIZON_LABELS = ['1h', '3h', '6h', '24h']


# ════════════════════════════════════════════════════════════════════════════
# HELPER: Build horizon-aggregated metrics
# ════════════════════════════════════════════════════════════════════════════
def get_overall_by_horizon(variable, metric_hmm, metric_mrf):
    """Extract the OVERALL (zone_id == 0) row per horizon."""
    df = df_metrics[(df_metrics['variable'] == variable) & (df_metrics['zone_id'] == 0)].copy()
    hmm_vals, mrf_vals = [], []
    for hz in HORIZONS:
        row = df[df['horizon'] == hz]
        if len(row) > 0:
            hmm_vals.append(float(row[metric_hmm].iloc[0]))
            mrf_vals.append(float(row[metric_mrf].iloc[0]))
        else:
            hmm_vals.append(np.nan)
            mrf_vals.append(np.nan)
    return hmm_vals, mrf_vals


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — AQI MAE: HMM vs HMM+MRF
# ════════════════════════════════════════════════════════════════════════════
print("\n[1] AQI MAE — HMM vs HMM+MRF")
hmm_mae, mrf_mae = get_overall_by_horizon('AQI', 'hmm_mae', 'mrf_mae')

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(HORIZONS))
w = 0.35
b1 = ax.bar(x - w/2, hmm_mae, w, label='HMM Only', color=BLUE_HMM, alpha=0.85, zorder=3)
b2 = ax.bar(x + w/2, mrf_mae, w, label='HMM + MRF', color=TEAL_MRF, alpha=0.85, zorder=3)

# Value labels
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, color='#c9d1d9')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, color='#c9d1d9')

ax.set_xticks(x)
ax.set_xticklabels(HORIZON_LABELS)
ax.set_xlabel('Forecast Horizon')
ax.set_ylabel('Mean Absolute Error (AQI units)')
ax.set_title('AQI Forecast Error: HMM vs HMM+MRF\nOverall (monitored zones aggregated)', pad=12)
ax.legend(loc='upper left')
subtitle_text(ax, 'Source: test.csv (Sep 14 – Dec 31, 2025) · Overall = aggregate of 7 monitored AQI zones')
savefig('aqi_mae_hmm_vs_mrf.png', fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — AQI RMSE: HMM vs HMM+MRF
# ════════════════════════════════════════════════════════════════════════════
print("[2] AQI RMSE — HMM vs HMM+MRF")
hmm_rmse, mrf_rmse = get_overall_by_horizon('AQI', 'hmm_rmse', 'mrf_rmse')

fig, ax = plt.subplots(figsize=(7, 4))
b1 = ax.bar(x - w/2, hmm_rmse, w, label='HMM Only', color=BLUE_HMM, alpha=0.85, zorder=3)
b2 = ax.bar(x + w/2, mrf_rmse, w, label='HMM + MRF', color=TEAL_MRF, alpha=0.85, zorder=3)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, color='#c9d1d9')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, color='#c9d1d9')
ax.set_xticks(x)
ax.set_xticklabels(HORIZON_LABELS)
ax.set_xlabel('Forecast Horizon')
ax.set_ylabel('Root Mean Squared Error (AQI units)')
ax.set_title('AQI Forecast RMSE: HMM vs HMM+MRF\nOverall (monitored zones aggregated)', pad=12)
ax.legend(loc='upper left')
subtitle_text(ax, 'Source: test.csv (Sep 14 – Dec 31, 2025) · Overall = aggregate of 7 monitored AQI zones')
savefig('aqi_rmse_hmm_vs_mrf.png', fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Rainfall MAE: HMM vs HMM+MRF
# ════════════════════════════════════════════════════════════════════════════
print("[3] Rainfall MAE — HMM vs HMM+MRF")
hmm_mae_rf, mrf_mae_rf = get_overall_by_horizon('Rainfall', 'hmm_mae', 'mrf_mae')

fig, ax = plt.subplots(figsize=(7, 4))
b1 = ax.bar(x - w/2, hmm_mae_rf, w, label='HMM Only', color=BLUE_HMM, alpha=0.85, zorder=3)
b2 = ax.bar(x + w/2, mrf_mae_rf, w, label='HMM + MRF', color=TEAL_MRF, alpha=0.85, zorder=3)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color='#c9d1d9')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color='#c9d1d9')
ax.set_xticks(x)
ax.set_xticklabels(HORIZON_LABELS)
ax.set_xlabel('Forecast Horizon')
ax.set_ylabel('Mean Absolute Error (mm)')
ax.set_title('Rainfall Forecast MAE: HMM vs HMM+MRF\nOverall (4 monitored rainfall zones aggregated)', pad=12)
ax.legend(loc='upper left')

# Note: MRF shows identical values to HMM for Rainfall (beta = 0 selected by calibration)
ax.text(0.98, 0.95,
        'Note: HMM+MRF values equal HMM\n(calibrated \u03b2 = 0 for Rainfall)',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=8, color='#8b949e', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1c2128', edgecolor='#30363d'))
subtitle_text(ax, 'Source: test.csv (Sep 14 – Dec 31, 2025) · Zones 4, 5, 10, 14')
savefig('rainfall_mae_hmm_vs_mrf.png', fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Rainfall RMSE: HMM vs HMM+MRF
# ════════════════════════════════════════════════════════════════════════════
print("[4] Rainfall RMSE — HMM vs HMM+MRF")
hmm_rmse_rf, mrf_rmse_rf = get_overall_by_horizon('Rainfall', 'hmm_rmse', 'mrf_rmse')

fig, ax = plt.subplots(figsize=(7, 4))
b1 = ax.bar(x - w/2, hmm_rmse_rf, w, label='HMM Only', color=BLUE_HMM, alpha=0.85, zorder=3)
b2 = ax.bar(x + w/2, mrf_rmse_rf, w, label='HMM + MRF', color=TEAL_MRF, alpha=0.85, zorder=3)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color='#c9d1d9')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color='#c9d1d9')
ax.set_xticks(x)
ax.set_xticklabels(HORIZON_LABELS)
ax.set_xlabel('Forecast Horizon')
ax.set_ylabel('Root Mean Squared Error (mm)')
ax.set_title('Rainfall Forecast RMSE: HMM vs HMM+MRF\nOverall (4 monitored rainfall zones aggregated)', pad=12)
ax.legend(loc='upper left')
ax.text(0.98, 0.95,
        'Note: HMM+MRF values equal HMM\n(calibrated \u03b2 = 0 for Rainfall)',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=8, color='#8b949e', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1c2128', edgecolor='#30363d'))
subtitle_text(ax, 'Source: test.csv (Sep 14 – Dec 31, 2025) · Zones 4, 5, 10, 14')
savefig('rainfall_rmse_hmm_vs_mrf.png', fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — AQI Zone-wise MAE (1h)
# ════════════════════════════════════════════════════════════════════════════
print("[5] AQI zone-wise MAE (1h)")
df_aqi_1h = df_metrics[(df_metrics['variable'] == 'AQI') &
                        (df_metrics['horizon'] == '1h') &
                        (df_metrics['zone_id'] != 0)].copy()
df_aqi_1h = df_aqi_1h.sort_values('zone_id')

fig, ax = plt.subplots(figsize=(9, 4.5))
zones = df_aqi_1h['zone_label'].tolist()
zx = np.arange(len(zones))
b1 = ax.bar(zx - w/2, df_aqi_1h['hmm_mae'], w, label='HMM Only', color=BLUE_HMM, alpha=0.85, zorder=3)
b2 = ax.bar(zx + w/2, df_aqi_1h['mrf_mae'], w, label='HMM + MRF', color=TEAL_MRF, alpha=0.85, zorder=3)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7.5, color='#c9d1d9')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7.5, color='#c9d1d9')
ax.set_xticks(zx)
ax.set_xticklabels(zones, rotation=30, ha='right')
ax.set_xlabel('Zone')
ax.set_ylabel('MAE (AQI units)')
ax.set_title('AQI 1h Forecast MAE by Zone: HMM vs HMM+MRF\n(7 Monitored Zones — Test Period)', pad=12)
ax.legend()
subtitle_text(ax, 'Source: test.csv (Sep 14 – Dec 31, 2025) · 1h ahead prediction')
savefig('aqi_zonewise_mae.png', fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — AQI Forecast Performance by Horizon (line chart)
# ════════════════════════════════════════════════════════════════════════════
print("[6] AQI horizon performance")
df_aqi_overall = df_metrics[(df_metrics['variable'] == 'AQI') &
                             (df_metrics['zone_id'] == 0)].copy()
df_aqi_overall = df_aqi_overall.set_index('horizon').reindex(HORIZONS)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (metric_hmm, metric_mrf, ylabel, title) in zip(axes, [
    ('hmm_mae', 'mrf_mae', 'MAE (AQI units)', 'Mean Absolute Error'),
    ('hmm_rmse', 'mrf_rmse', 'RMSE (AQI units)', 'Root Mean Squared Error'),
]):
    hmm_vals = df_aqi_overall[metric_hmm].values
    mrf_vals = df_aqi_overall[metric_mrf].values
    hz_x = np.arange(len(HORIZONS))

    ax.plot(hz_x, hmm_vals, 'o-', color=BLUE_HMM, linewidth=2, markersize=7,
            label='HMM Only', zorder=4)
    ax.plot(hz_x, mrf_vals, 's--', color=TEAL_MRF, linewidth=2, markersize=7,
            label='HMM + MRF', zorder=4)
    ax.fill_between(hz_x, hmm_vals, mrf_vals,
                    alpha=0.12, color=TEAL_MRF)

    for xi, (h, m) in enumerate(zip(hmm_vals, mrf_vals)):
        ax.annotate(f'{h:.1f}', (xi, h), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=8, color=BLUE_HMM)
        ax.annotate(f'{m:.1f}', (xi, m), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=8, color=TEAL_MRF)

    ax.set_xticks(hz_x)
    ax.set_xticklabels(HORIZON_LABELS)
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left')

fig.suptitle('AQI Forecast Performance vs Forecast Horizon\n(Overall, 7 Monitored Zones, Test Period)',
             y=1.02, fontsize=13)
fig.tight_layout()
subtitle_text(axes[1], 'Source: test.csv (Sep 14 – Dec 31, 2025)')
savefig('aqi_horizon_performance.png', fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Alert Distribution
# ════════════════════════════════════════════════════════════════════════════
print("[7] Alert distribution")

# Compute counts by variable and status (collapse rare statuses)
STATUS_RENAME = {
    'WARNING': 'WARNING',
    'WATCH': 'WATCH',
    'NORMAL': 'NORMAL',
    'UNMONITORED_INSUFFICIENT_DATA': 'Unmonitored / Insufficient Data',
    'UNAVAILABLE_MISSING_CLIMATOLOGY': 'Unavailable — Missing Climatology',
}
COLORS_BY_STATUS = {
    'WARNING': RED_WARN,
    'WATCH': AMBER_WATCH,
    'NORMAL': GREEN_NORM,
    'Unmonitored / Insufficient Data': GRAY_UNMON,
    'Unavailable — Missing Climatology': '#374151',
}

df_a = df_alerts.copy()
df_a['status_display'] = df_a['alert_status'].map(STATUS_RENAME)

# Separate AQI and Rainfall
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, var in zip(axes, ['AQI', 'Rainfall']):
    sub = df_a[df_a['variable'] == var]
    if len(sub) == 0:
        ax.text(0.5, 0.5, f'No {var} alert data', ha='center', va='center',
                transform=ax.transAxes, color='#8b949e')
        continue

    counts = sub.groupby('status_display').size().reset_index(name='count')
    counts['pct'] = counts['count'] / counts['count'].sum() * 100

    order_keys = ['WARNING', 'WATCH', 'NORMAL',
                  'Unmonitored / Insufficient Data', 'Unavailable \u2014 Missing Climatology']
    present_keys = [k for k in order_keys if k in counts['status_display'].values]
    counts = counts.set_index('status_display').reindex(present_keys).reset_index()
    counts = counts.dropna(subset=['count'])
    counts = counts[counts['count'] > 0]

    if len(counts) == 0:
        ax.text(0.5, 0.5, f'No {var} alert data', ha='center', va='center',
                transform=ax.transAxes, color='#8b949e')
        continue

    colors = [COLORS_BY_STATUS.get(s, GRAY_UNMON) for s in counts['status_display']]
    bars = ax.barh(counts['status_display'], counts['count'], color=colors, alpha=0.85, zorder=3)
    max_count = float(counts['count'].max())

    for bar, (_, row) in zip(bars, counts.iterrows()):
        ax.text(bar.get_width() + max_count * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(row['count']):,}  ({row['pct']:.1f}%)",
                va='center', ha='left', fontsize=8.5, color='#c9d1d9')

    ax.set_xlabel('Alert Count')
    ax.set_title(f'{var} Alert Distribution\n(Test Period: Sep 14 \u2013 Dec 31, 2025)', pad=10)
    ax.set_xlim(0, max_count * 1.30)
    ax.invert_yaxis()

fig.suptitle('UrbanPulse Alert Status Distribution \u2014 All Test-Period Alerts', fontsize=13, y=1.01)
fig.tight_layout()
savefig('alert_distribution.png', fig)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — AQI Threshold Confusion Matrix
# ════════════════════════════════════════════════════════════════════════════
print("[8] AQI threshold confusion matrix")

# Binary labels: threshold = AQI > 200
# Only monitored zones with actual observations (no fabrication)
df_aqi = df_preds[(df_preds['variable'] == 'AQI') &
                  (df_preds['horizon'] == '1h') &
                  (df_preds['is_monitored'] == True) &
                  (df_preds['actual'].notna()) &
                  (df_preds['mrf_pred'].notna())].copy()

AQI_THRESH = 200.0
y_true = (df_aqi['actual'] > AQI_THRESH).astype(int).values
y_pred = (df_aqi['mrf_pred'] > AQI_THRESH).astype(int).values

# Confusion matrix
TN = int(((y_true == 0) & (y_pred == 0)).sum())
FP = int(((y_true == 0) & (y_pred == 1)).sum())
FN = int(((y_true == 1) & (y_pred == 0)).sum())
TP = int(((y_true == 1) & (y_pred == 1)).sum())

cm = np.array([[TN, FP], [FN, TP]])
n_total = len(y_true)
accuracy = (TP + TN) / n_total
precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                          gridspec_kw={'width_ratios': [1.1, 1]})

# Confusion matrix heatmap
ax_cm = axes[0]
cmap = LinearSegmentedColormap.from_list('cm_cmap', ['#161b22', '#312e81', '#6366f1'], N=256)
im = ax_cm.imshow(cm, cmap=cmap, aspect='auto')

labels = [['TN', 'FP'], ['FN', 'TP']]
cell_colors = [['#22c55e', '#ef4444'], ['#f59e0b', '#06b6d4']]
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = val / n_total * 100
        ax_cm.text(j, i, f'{labels[i][j]}\n{val:,}\n({pct:.1f}%)',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color=cell_colors[i][j])

ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['Pred: AQI ≤ 200', 'Pred: AQI > 200'], fontsize=9)
ax_cm.set_yticklabels(['Act: AQI ≤ 200', 'Act: AQI > 200'], fontsize=9)
ax_cm.set_title('AQI Threshold Confusion Matrix\n(MRF Predicted vs Actual, 1h Horizon)', pad=10)

# Metrics panel
ax_metrics = axes[1]
ax_metrics.set_facecolor('#161b22')
ax_metrics.set_xlim(0, 1)
ax_metrics.set_ylim(0, 1)
ax_metrics.axis('off')

metrics_text = [
    ('Definition', f'Binary: AQI > {AQI_THRESH:.0f} (CPCB Poor)', '#a78bfa'),
    ('Source', '7 monitored AQI zones, 1h horizon', '#8b949e'),
    ('Test samples', f'{n_total:,}', '#c9d1d9'),
    ('Actual positives', f'{TP + FN:,}  ({(TP+FN)/n_total:.1%})', '#c9d1d9'),
    ('', '', ''),
    ('Accuracy', f'{accuracy:.3f}', '#22c55e'),
    ('Precision', f'{precision:.3f}', '#06b6d4'),
    ('Recall (Sensitivity)', f'{recall:.3f}', AMBER_WATCH),
    ('F1 Score', f'{f1:.3f}', ACCENT),
    ('', '', ''),
    ('True Negatives', f'{TN:,}', '#22c55e'),
    ('False Positives', f'{FP:,}', RED_WARN),
    ('False Negatives', f'{FN:,}', AMBER_WATCH),
    ('True Positives', f'{TP:,}', '#06b6d4'),
]

y_pos = 0.95
for label, value, color in metrics_text:
    if label == '':
        y_pos -= 0.03
        continue
    ax_metrics.text(0.02, y_pos, f'{label}:', fontsize=8.5, color='#8b949e', va='top')
    ax_metrics.text(0.98, y_pos, value, fontsize=8.5, color=color, va='top', ha='right', fontweight='600')
    y_pos -= 0.065

ax_metrics.text(0.5, 0.03,
    'Note: threshold applied to point prediction (MRF mean)\nnot probabilistic P(AQI > 200)',
    fontsize=7.5, color='#6b7280', ha='center', style='italic', va='bottom')

fig.suptitle('AQI > 200 Threshold Classification Performance\nTest Period: Sep 14 – Dec 31, 2025',
             fontsize=12, y=1.01)
fig.tight_layout()
savefig('aqi_confusion_matrix.png', fig)

print(f"   AQI CM: TN={TN:,} FP={FP:,} FN={FN:,} TP={TP:,}")
print(f"   Accuracy={accuracy:.3f} Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Rainfall 24h Confusion Matrix
# ════════════════════════════════════════════════════════════════════════════
print("[9] Rainfall 24h accumulation confusion matrix")

# Compute actual 24h rolling accumulation from test.csv
# Only for the 4 monitored rainfall zones: 4, 5, 10, 14
MONITORED_RF = [4, 5, 10, 14]
RF_THRESH = 64.5  # mm / 24h

df_test['timestamp'] = pd.to_datetime(df_test['timestamp'], utc=True)
df_rf_test = df_test[df_test['zone_id'].isin(MONITORED_RF)][
    ['timestamp', 'zone_id', 'rainfall']
].copy()
df_rf_test = df_rf_test[df_rf_test['rainfall'].notna()]
df_rf_test = df_rf_test.sort_values(['zone_id', 'timestamp'])

# Rolling 24h accumulation per zone
df_rf_test['acc_24h_actual'] = df_rf_test.groupby('zone_id')['rainfall'].transform(
    lambda s: s.rolling(window=24, min_periods=24).sum()
)
df_rf_valid = df_rf_test[df_rf_test['acc_24h_actual'].notna()].copy()

# Get predicted 24h accumulation from alerts (forecast_horizon == '24h_accumulation')
df_alerts_rf = df_alerts[
    (df_alerts['variable'] == 'Rainfall') &
    (df_alerts['forecast_horizon'] == '24h_accumulation') &
    (df_alerts['zone_id'].isin(MONITORED_RF)) &
    (df_alerts['predicted_value'].notna())
].copy()
df_alerts_rf['timestamp'] = pd.to_datetime(df_alerts_rf['timestamp'], utc=True)
df_alerts_rf = df_alerts_rf[['timestamp', 'zone_id', 'predicted_value']].rename(
    columns={'predicted_value': 'acc_24h_pred'}
)

# Merge on timestamp + zone_id
df_rf_merged = df_rf_valid.merge(df_alerts_rf, on=['timestamp', 'zone_id'], how='inner')
print(f"   Merged RF rows: {len(df_rf_merged):,}")

if len(df_rf_merged) > 0:
    y_true_rf = (df_rf_merged['acc_24h_actual'] >= RF_THRESH).astype(int).values
    y_pred_rf = (df_rf_merged['acc_24h_pred'] >= RF_THRESH).astype(int).values

    TN_r = int(((y_true_rf == 0) & (y_pred_rf == 0)).sum())
    FP_r = int(((y_true_rf == 0) & (y_pred_rf == 1)).sum())
    FN_r = int(((y_true_rf == 1) & (y_pred_rf == 0)).sum())
    TP_r = int(((y_true_rf == 1) & (y_pred_rf == 1)).sum())

    cm_r = np.array([[TN_r, FP_r], [FN_r, TP_r]])
    n_r = len(y_true_rf)
    acc_r = (TP_r + TN_r) / n_r
    prec_r = TP_r / (TP_r + FP_r) if (TP_r + FP_r) > 0 else np.nan
    rec_r = TP_r / (TP_r + FN_r) if (TP_r + FN_r) > 0 else np.nan
    f1_r = 2 * prec_r * rec_r / (prec_r + rec_r) if (prec_r + rec_r) > 0 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                              gridspec_kw={'width_ratios': [1.1, 1]})
    ax_cm = axes[0]
    im = ax_cm.imshow(cm_r, cmap=cmap, aspect='auto')

    for i in range(2):
        for j in range(2):
            val = cm_r[i, j]
            pct = val / n_r * 100
            ax_cm.text(j, i, f'{labels[i][j]}\n{val:,}\n({pct:.1f}%)',
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       color=cell_colors[i][j])

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Pred: < 64.5 mm', 'Pred: ≥ 64.5 mm'], fontsize=9)
    ax_cm.set_yticklabels(['Act: < 64.5 mm', 'Act: ≥ 64.5 mm'], fontsize=9)
    ax_cm.set_title('Rainfall 24h Accumulation Threshold Confusion Matrix\n(HMM Predicted vs Actual Rolling 24h)', pad=10)

    ax_metrics = axes[1]
    ax_metrics.set_facecolor('#161b22')
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.axis('off')

    rf_metrics = [
        ('Definition', f'24h accumulation >= {RF_THRESH} mm (IMD Heavy)', '#a78bfa'),
        ('Source', 'Zones 4, 5, 10, 14 (rolling 24h actual)', '#8b949e'),
        ('Test samples', f'{n_r:,}', '#c9d1d9'),
        ('Actual positives', f'{TP_r + FN_r:,}  ({(TP_r+FN_r)/n_r:.1%})', '#c9d1d9'),
        ('', '', ''),
        ('Accuracy', f'{acc_r:.3f}', '#22c55e'),
        ('Precision', f'{prec_r:.3f}' if pd.notna(prec_r) else 'N/A', '#06b6d4'),
        ('Recall (Sensitivity)', f'{rec_r:.3f}' if pd.notna(rec_r) else 'N/A', AMBER_WATCH),
        ('F1 Score', f'{f1_r:.3f}' if pd.notna(f1_r) else 'N/A', ACCENT),
        ('', '', ''),
        ('True Negatives', f'{TN_r:,}', '#22c55e'),
        ('False Positives', f'{FP_r:,}', RED_WARN),
        ('False Negatives', f'{FN_r:,}', AMBER_WATCH),
        ('True Positives', f'{TP_r:,}', '#06b6d4'),
    ]
    y_pos = 0.95
    for label, value, color in rf_metrics:
        if label == '':
            y_pos -= 0.03
            continue
        ax_metrics.text(0.02, y_pos, f'{label}:', fontsize=8.5, color='#8b949e', va='top')
        ax_metrics.text(0.98, y_pos, value, fontsize=8.5, color=color, va='top', ha='right', fontweight='600')
        y_pos -= 0.065

    ax_metrics.text(0.5, 0.03,
        'Actual: rolling 24h sum of hourly rainfall (zone-level).\n'
        'Predicted: HMM 24h forecast accumulation mean.\n'
        '64.5 mm / 24h is the IMD Heavy Rainfall official threshold.',
        fontsize=7.5, color='#6b7280', ha='center', style='italic', va='bottom')

    fig.suptitle('Rainfall 24h Accumulation Threshold Classification\nTest Period: Sep 14 – Dec 31, 2025',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    savefig('rainfall_confusion_matrix.png', fig)

    print(f"   RF CM: TN={TN_r:,} FP={FP_r:,} FN={FN_r:,} TP={TP_r:,}")
    print(f"   Accuracy={acc_r:.3f} Precision={prec_r:.3f} Recall={rec_r:.3f} F1={f1_r:.3f}")
else:
    print("   WARNING: No merged RF data — skipping confusion matrix.")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Temperature Evaluation Unavailable
# ════════════════════════════════════════════════════════════════════════════
print("[10] Temperature evaluation unavailable notice")

fig, ax = plt.subplots(figsize=(9, 4))
ax.set_facecolor('#161b22')
ax.axis('off')

# Title block
ax.text(0.5, 0.92, 'Temperature Test-Set Evaluation — Unavailable',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=14, fontweight='bold', color='#e6edf3')

# Horizontal rule
ax.plot([0.05, 0.95], [0.83, 0.83],
        color='#6366f1', linewidth=1, alpha=0.3, transform=ax.transAxes)

lines = [
    ('Reason', '#a78bfa', 12, 'bold'),
    ('', '', 10, 'normal'),
    ('The available temperature observations end on 22 June 2024,', '#c9d1d9', 10, 'normal'),
    ('while the final model test period covers September 14 – December 31, 2025.', '#c9d1d9', 10, 'normal'),
    ('', '', 10, 'normal'),
    ('There are zero non-null temperature observation values in test.csv.', '#c9d1d9', 10, 'normal'),
    ('Consequently, no empirical test-set evaluation can be performed.', '#c9d1d9', 10, 'normal'),
    ('', '', 10, 'normal'),
    ('Implication for Alert Engine', '#a78bfa', 10, 'bold'),
    ('', '', 10, 'normal'),
    ('The IMD Coastal Heatwave alert also requires a verified climatological', '#c9d1d9', 10, 'normal'),
    ('normal (Tmax), which is not present in the dataset and was not fabricated.', '#c9d1d9', 10, 'normal'),
    ('All temperature alerts are labeled: UNAVAILABLE — MISSING CLIMATOLOGY.', '#f59e0b', 10, 'normal'),
    ('', '', 10, 'normal'),
    ('No accuracy, MAE, RMSE, or confusion matrix is reported for temperature.', '#ef4444', 10, 'bold'),
]

y_pos = 0.78
for text, color, size, weight in lines:
    if text == '':
        y_pos -= 0.025
        continue
    ax.text(0.5, y_pos, text, transform=ax.transAxes,
            ha='center', va='top', fontsize=size,
            color=color, fontweight=weight)
    y_pos -= 0.072

fig.suptitle('UrbanPulse — Temperature Model Test Evaluation',
             fontsize=13, color='#e6edf3', y=0.98)
savefig('temperature_evaluation_unavailable.png', fig)


print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print(f"Output directory: {FIGURES_DIR}")
print("=" * 60)
for f in sorted(os.listdir(FIGURES_DIR)):
    size = os.path.getsize(os.path.join(FIGURES_DIR, f))
    print(f"  {f:<50s}  {size/1024:.1f} KB")
