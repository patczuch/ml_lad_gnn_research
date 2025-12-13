import os.path as osp
import pandas as pd
from typing import List

METRICS_DEFAULT = [
    # 'num_epochs',
    'test_avg_acc',
    'AUC',
    'F1',
    'ECE',
    'Brier',
]

# Rename map for display
METRIC_RENAME = {
    # 'num_epochs': 'Epochs',
    'test_avg_acc': 'Accuracy',
    'AUC': 'AUC',
    'F1': 'F1',
    'ECE': 'ECE',
    'Brier': 'Brier',
}

BACKBONES = ['GAT', 'GCN', 'GIN', 'GraphSAGE', 'SAGE']
MODEL_ORDER = [f'{m}{s}' for m in BACKBONES for s in ['', '+LAD']]


def load_summary(path: str) -> pd.DataFrame:
    if not osp.exists(path):
        raise FileNotFoundError(f"summary CSV not found: {path}")
    df = pd.read_csv(path, sep=';')
    df.columns = [c.strip() for c in df.columns]

    required = ['dataset', 'type', 'train_mode'] + METRICS_DEFAULT
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {missing}")

    for col in ['test_avg_acc', 'test_std_acc', 'AUC', 'F1', 'ECE', 'Brier', 'num_epochs']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['train_mode'] = df['train_mode'].astype(str).str.upper().str.strip()
    df = df[df['train_mode'].isin(['P', 'S'])].copy()
    return df


def aggregate_ps(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    agg = df.groupby(['dataset', 'type', 'train_mode'])[metrics].agg(['mean', 'std', 'count'])
    out = pd.DataFrame()

    for metric in metrics:
        mdf = agg[metric].reset_index()
        pivot_mean = mdf.pivot(index=['dataset', 'type'], columns='train_mode', values='mean')
        pivot_std = mdf.pivot(index=['dataset', 'type'], columns='train_mode', values='std')

        for mode in ['P', 'S']:
            if mode not in pivot_mean.columns:
                pivot_mean[mode] = pd.NA
                pivot_std[mode] = pd.NA

        metric_name = METRIC_RENAME.get(metric, metric)

        out[f'{metric_name}_P'] = (
            pivot_mean['P'].map(lambda x: '' if pd.isna(x) else f"{x:.4f}") +
            r" $\pm$ " +
            pivot_std['P'].map(lambda x: '' if pd.isna(x) else f"{x:.4f}")
        )
        out[f'{metric_name}_S'] = (
            pivot_mean['S'].map(lambda x: '' if pd.isna(x) else f"{x:.4f}") +
            r" $\pm$ " +
            pivot_std['S'].map(lambda x: '' if pd.isna(x) else f"{x:.4f}")
        )

    out = out.reset_index()
    return out


def reshape_all_models(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        base = r['type']
        if base not in BACKBONES:
            continue
        if base == "GraphSAGE":
            base = "SAGE"

        row_base = {'dataset': r['dataset'], 'Model': base}
        row_lad = {'dataset': r['dataset'], 'Model': f'{base}+LAD'}

        for metric in metrics:
            metric_name = METRIC_RENAME.get(metric, metric)
            row_base[metric_name] = r[f'{metric_name}_P']
            row_lad[metric_name] = r[f'{metric_name}_S']

        rows.append(row_base)
        rows.append(row_lad)

    out = pd.DataFrame(rows)
    out['Model'] = pd.Categorical(out['Model'], categories=MODEL_ORDER, ordered=True)
    out = out.sort_values(['dataset', 'Model']).reset_index(drop=True)
    return out


def build_latex_tables_per_dataset(df: pd.DataFrame, metrics: List[str]) -> str:
    agg_df = aggregate_ps(df, metrics)
    final_df = reshape_all_models(agg_df, metrics)

    latex_str = '% Auto-generated LaTeX tables per dataset\n\n'
    datasets = final_df['dataset'].unique()

    for dataset in datasets:
        ddf = final_df[final_df['dataset'] == dataset].drop(columns=['dataset'])
        latex_str += f"\\begin{{table}}[ht]\n\\centering\n\\small\n\\caption{{Results for {dataset}}}\n"
        latex_str += ddf.to_latex(index=False, escape=False)
        latex_str += "\\end{table}\n\n"

    return latex_str


# Example usage
metrics_to_use = METRICS_DEFAULT
df_summary = load_summary("results/summary.csv")
latex_content = build_latex_tables_per_dataset(df_summary, metrics_to_use)

with open("results/summary.tex", 'w', encoding='utf-8') as f:
    f.write(latex_content)
