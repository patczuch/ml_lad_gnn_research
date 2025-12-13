import argparse
import os
import os.path as osp
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_epoch_logs(input_dir: str) -> List[str]:
    files = []
    if not osp.isdir(input_dir):
        return files
    # Expect structure: input_dir/<TYPE>/*.csv
    for root, dirs, fnames in os.walk(input_dir):
        for fn in fnames:
            if fn.lower().endswith('.csv'):
                files.append(osp.join(root, fn))
    return sorted(files)


def read_epoch_log(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Reads a single epoch log file created by main.py per-epoch logging.
    Returns (df, meta) where df has at least columns: epoch, val_loss and meta has dataset,type,train_mode.
    """
    # Extract metadata from first comment lines beginning with '#'
    meta = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for _ in range(4):
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith('#'):
                    m = re.match(r"#\s*(\w+):\s*(.*)\s*$", line)
                    if m:
                        key = m.group(1).strip().lower()
                        val = m.group(2).strip()
                        meta[key] = val
                else:
                    # rewind to start of header
                    f.seek(pos)
                    break
    except Exception:
        pass

    # Fallback metadata from filename: <DATASET>_<TYPE>_<TRAINMODE>_TIMESTAMP.csv
    base = osp.basename(path)
    stem = osp.splitext(base)[0]
    parts = stem.split('_')
    if ('dataset' not in meta or 'type' not in meta or 'train_mode' not in meta) and len(parts) >= 4:
        # last part is timestamp which may contain dash; join the rest
        # dataset may contain dashes; assume the last 2 before timestamp are TYPE and TRAINMODE
        # Safer: pick from the end
        train_mode = parts[-2]
        model_type = parts[-3]
        dataset = '_'.join(parts[:-3])
        meta.setdefault('dataset', dataset)
        meta.setdefault('type', model_type)
        meta.setdefault('train_mode', train_mode)

    # Read CSV, semicolon separated, comments starting with '#'
    df = pd.read_csv(path, sep=';', comment='#')
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    # Ensure required columns
    if 'epoch' not in df.columns or 'val_loss' not in df.columns:
        raise ValueError(f"Missing required columns in epoch log: {path}")
    # Coerce numeric
    for col in ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df, meta


def aggregate_runs(dfs: List[pd.DataFrame], use_min_common: bool = True) -> pd.DataFrame:
    """
    Aggregate multiple run DataFrames (each with columns epoch, val_loss) by epoch.
    Returns a DataFrame with columns: epoch, mean, std, n.
    If use_min_common is True, restrict to epochs that are present in all runs (intersection).
    Otherwise, compute stats per-epoch over available runs (union).
    """
    if not dfs:
        return pd.DataFrame(columns=['epoch', 'mean', 'std', 'n'])

    # Align by epoch
    counts = defaultdict(int)
    values_by_epoch = defaultdict(list)
    epoch_sets = []
    for df in dfs:
        cur = df[['epoch', 'val_loss']].dropna()
        epoch_sets.append(set(cur['epoch'].astype(int).tolist()))
        for e, v in zip(cur['epoch'].astype(int), cur['val_loss'].astype(float)):
            values_by_epoch[int(e)].append(float(v))
            counts[int(e)] += 1

    if use_min_common:
        common = set.intersection(*epoch_sets) if epoch_sets else set()
        epochs = sorted(common)
    else:
        epochs = sorted(values_by_epoch.keys())

    means, stds, ns = [], [], []
    for e in epochs:
        vals = values_by_epoch.get(e, [])
        if use_min_common:
            # keep only epochs with full coverage
            if counts[e] != len(dfs):
                continue
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ns.append(len(vals))

    return pd.DataFrame({'epoch': epochs[:len(means)], 'mean': means, 'std': stds, 'n': ns})


def plot_group(dataset: str, model_type: str, agg_P: pd.DataFrame, agg_S: pd.DataFrame, out_path: str,
               title_extra: str = ''):
    plt.figure(figsize=(8, 5))

    def plot_one(agg: pd.DataFrame, label: str, color: str):
        if agg is None or agg.empty:
            return
        x = agg['epoch'].to_numpy()
        y = agg['mean'].to_numpy()
        s = agg['std'].to_numpy()
        plt.plot(x, y, label=label, color=color, linewidth=2)
        # Std as shaded band and thin boundary lines
        plt.fill_between(x, y - s, y + s, color=color, alpha=0.15, linewidth=0)
        plt.plot(x, y - s, color=color, alpha=0.4, linewidth=0.8, linestyle='--')
        plt.plot(x, y + s, color=color, alpha=0.4, linewidth=0.8, linestyle='--')

    plot_one(agg_P, 'Pure', '#1f77b4')
    plot_one(agg_S, 'LAD', '#d62728')

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    title = f"{dataset} — {model_type} Validation Loss"
    if title_extra:
        title += f" {title_extra}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Aggregate epoch logs and plot mean±std validation loss for P and S')
    parser.add_argument('--input_dir', type=str, default=osp.join('results', 'epoch_logs'),
                        help='Directory containing epoch log CSVs grouped by type subfolders')
    parser.add_argument('--out_dir', type=str, default=osp.join('results', 'plots', 'val_loss'),
                        help='Directory to save plots')
    parser.add_argument('--min_common', action='store_true',
                        help='Use only epochs common to all runs in a group (default: off)')
    parser.add_argument('--save_csv', action='store_true',
        help='Also save aggregated CSVs (epoch, mean, std, n) next to the plots')

    args = parser.parse_args()

    files = find_epoch_logs(args.input_dir)
    if not files:
        print(f"[WARN] No epoch logs found under: {args.input_dir}")
        return

    # Group logs by (dataset, type, train_mode)
    groups: Dict[Tuple[str, str, str], List[pd.DataFrame]] = defaultdict(list)
    for fp in files:
        try:
            df, meta = read_epoch_log(fp)
            dataset = meta.get('dataset', '').strip()
            model_type = meta.get('type', '').strip()
            train_mode = meta.get('train_mode', '').strip().upper()
            if train_mode not in {'P', 'S'}:
                continue
            if not dataset or not model_type:
                continue
            groups[(dataset, model_type, train_mode)].append(df)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")

    # For each (dataset, type) combine P and S
    pairs = defaultdict(dict)
    for (dataset, model_type, mode), dfs in groups.items():
        if not dfs:
            continue
        agg = aggregate_runs(dfs, use_min_common=args.min_common)
        pairs[(dataset, model_type)][mode] = agg

    # Plot
    for (dataset, model_type), modes in sorted(pairs.items()):
        agg_P = modes.get('P', pd.DataFrame())
        agg_S = modes.get('S', pd.DataFrame())
        if (agg_P is None or agg_P.empty) and (agg_S is None or agg_S.empty):
            continue
        out_name = f"{dataset}_{model_type}_val_loss.png"
        out_path = osp.join(args.out_dir, out_name)
        title_extra = "(min common epochs)" if args.min_common else ""
        plot_group(dataset, model_type, agg_P, agg_S, out_path, title_extra=title_extra)

        if args.save_csv:
            base = osp.splitext(out_path)[0]
            if agg_P is not None and not agg_P.empty:
                agg_P.to_csv(base + '_P.csv', index=False)
            if agg_S is not None and not agg_S.empty:
                agg_S.to_csv(base + '_S.csv', index=False)

    print(f"Saved plots to: {osp.abspath(args.out_dir)}")


if __name__ == '__main__':
    main()
