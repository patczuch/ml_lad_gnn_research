import os.path as osp
import pandas as pd
from typing import List
import numpy as np

METRICS_DEFAULT = ['test_avg_acc', 'AUC', 'F1', 'ECE', 'Brier']
METRIC_RENAME = {
    'test_avg_acc': 'Accuracy',
    'AUC': 'AUC',
    'F1': 'F1',
    'ECE': 'ECE',
    'Brier': 'Brier',
}

# Metryki gdzie mniejsza wartość = lepsza
LOWER_IS_BETTER = ['ECE', 'Brier']

BACKBONES = ['GAT', 'GCN', 'GIN', 'GraphSAGE']
MODEL_ORDER = [f'{m}{s}' for m in BACKBONES for s in ['', '+LAD']]


def load_summary(path: str) -> pd.DataFrame:
    if not osp.exists(path):
        raise FileNotFoundError(f"summary CSV not found: {path}")
    df = pd.read_csv(path, sep=';', header=None)
    df.columns = ['dataset', 'type', 'train_mode', 'test_avg_acc', 'test_std_acc',
                  'AUC', 'F1', 'ECE', 'Brier', 'num_epochs']

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
        out[f'{metric_name}_P_mean'] = pivot_mean['P']
        out[f'{metric_name}_P_std'] = pivot_std['P']
        out[f'{metric_name}_S_mean'] = pivot_mean['S']
        out[f'{metric_name}_S_std'] = pivot_std['S']
        out[f'{metric_name}_P'] = (
                pivot_mean['P'].map(lambda x: '' if pd.isna(x) else f"{x:.4f}") +
                " ± " +
                pivot_std['P'].map(lambda x: '' if pd.isna(x) else f"{x:.4f}")
        )
        out[f'{metric_name}_S'] = (
                pivot_mean['S'].map(lambda x: '' if pd.isna(x) else f"{x:.4f}") +
                " ± " +
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
        display_base = "SAGE" if base == "GraphSAGE" else base

        row_base = {'dataset': r['dataset'], 'Model': display_base}
        row_lad = {'dataset': r['dataset'], 'Model': f'{display_base}+LAD'}
        for metric in metrics:
            metric_name = METRIC_RENAME.get(metric, metric)
            row_base[metric_name] = r[f'{metric_name}_P']
            row_base[f'{metric_name}_mean'] = r[f'{metric_name}_P_mean']
            row_lad[metric_name] = r[f'{metric_name}_S']
            row_lad[f'{metric_name}_mean'] = r[f'{metric_name}_S_mean']

        rows.append(row_base)
        rows.append(row_lad)

    out = pd.DataFrame(rows)
    model_order_sage = [m.replace('GraphSAGE', 'SAGE') for m in MODEL_ORDER]
    out['Model'] = pd.Categorical(out['Model'], categories=model_order_sage, ordered=True)
    out = out.sort_values(['dataset', 'Model']).reset_index(drop=True)
    return out


def compute_rankings(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Oblicza ranking dla każdego modelu w każdej metryce i datasecie."""
    rankings = []

    for dataset in df['dataset'].unique():
        ddf = df[df['dataset'] == dataset].copy()

        for metric in metrics:
            metric_name = METRIC_RENAME.get(metric, metric)
            mean_col = f'{metric_name}_mean'

            if mean_col not in ddf.columns:
                continue

            # Sortowanie - ascending dla ECE/Brier, descending dla reszty
            ascending = metric_name in LOWER_IS_BETTER
            ranked = ddf[['Model', mean_col]].dropna(subset=[mean_col]).copy()
            ranked['rank'] = ranked[mean_col].rank(ascending=ascending, method='min')

            for _, row in ranked.iterrows():
                rankings.append({
                    'dataset': dataset,
                    'Model': row['Model'],
                    'metric': metric_name,
                    'rank': row['rank']
                })

    return pd.DataFrame(rankings)


def compute_average_ranks(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """Oblicza średnią pozycję w rankingu dla każdego modelu."""
    avg_ranks = rankings_df.groupby('Model')['rank'].agg(['mean', 'std', 'count']).reset_index()
    avg_ranks.columns = ['Model', 'Średnia pozycja', 'Odch. std.', 'Liczba rankingów']
    avg_ranks = avg_ranks.sort_values('Średnia pozycja').reset_index(drop=True)
    return avg_ranks


def compute_average_ranks_per_metric(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """Oblicza średnią pozycję w rankingu dla każdego modelu per metryka."""
    pivot = rankings_df.pivot_table(
        index='Model',
        columns='metric',
        values='rank',
        aggfunc='mean'
    ).reset_index()

    # Dodaj średnią ze wszystkich metryk
    metric_cols = [c for c in pivot.columns if c != 'Model']
    pivot['Średnia'] = pivot[metric_cols].mean(axis=1)
    pivot = pivot.sort_values('Średnia').reset_index(drop=True)

    return pivot


def build_markdown_tables_per_dataset(df: pd.DataFrame, metrics: List[str]) -> str:
    agg_df = aggregate_ps(df, metrics)
    final_df = reshape_all_models(agg_df, metrics)

    # Oblicz rankingi
    rankings_df = compute_rankings(final_df, metrics)
    avg_ranks = compute_average_ranks(rankings_df)
    avg_ranks_per_metric = compute_average_ranks_per_metric(rankings_df)

    md_parts = ["# Wyniki eksperymentów\n\n"]

    # Tabela ze średnimi pozycjami w rankingu
    md_parts.append("## Średnia pozycja w rankingu (wszystkie datasety i metryki)\n\n")
    md_parts.append(avg_ranks.to_markdown(index=False, floatfmt=".2f"))
    md_parts.append("\n\n")

    # Tabela ze średnimi pozycjami per metryka
    md_parts.append("## Średnia pozycja w rankingu per metryka\n\n")
    md_parts.append(avg_ranks_per_metric.to_markdown(index=False, floatfmt=".2f"))
    md_parts.append("\n\n")

    # Tabele wyników per dataset
    md_parts.append("## Szczegółowe wyniki per dataset\n\n")
    datasets = final_df['dataset'].unique()

    display_cols = ['Model'] + [METRIC_RENAME.get(m, m) for m in metrics]

    for dataset in datasets:
        ddf = final_df[final_df['dataset'] == dataset][display_cols].copy()
        md_parts.append(f"### {dataset}\n\n")
        md_parts.append(ddf.to_markdown(index=False))
        md_parts.append("\n\n")

    return "".join(md_parts)


if __name__ == "__main__":
    metrics_to_use = METRICS_DEFAULT
    df_summary = load_summary("../results/summary.csv")
    markdown_content = build_markdown_tables_per_dataset(df_summary, metrics_to_use)
    with open("../results/results_comparison.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print("Zapisano wyniki do results/results_comparison.md")
