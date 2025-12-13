import argparse
import subprocess
import sys
import random
from datetime import datetime

"""
hyperparams for MUTAG, PROTEINS, COLLAB, IMDB-BINARY, IMDB-MULTI:
batch_size = 32
nhid = 64
nlayers = 2
gat_heads = 4
epochs = 500
learning rate = 0.001
dropout = 0.5
with_bn = true
with_bias = true
weight_decay = 5e-5
scheduler_patience = 50
scheduler_factor = 0.1
alpha = 1.0
tau = 0.1
early_stop = 7

hyperparams for REDDIT-BINARY
batch_size = 8
nhid = 64
nlayers = 2
gat_heads = 2
epochs = 500
learning rate = 0.001
dropout = 0.5
with_bn = false
with_bias = true
weight_decay = 5e-4
scheduler_patience = 10
scheduler_factor = 0.5
alpha = 0.3
tau = 0.3
early_stop = 15

hyperparams for ogbg-molhiv
batch_size = 256
nhid = 128
nlayers = 4
gat_heads = 4
epochs = 500
learning rate = 0.001
dropout = 0.2
with_bn = true
with_bias = true
weight_decay = 1e-3
scheduler_patience = 10
scheduler_factor = 0.5
alpha = 0.3
tau = 0.3
early_stop = 25
"""

# MUTAG, PROTEINS, COLLAB, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY, ogbg-molhiv

def main():
    parser = argparse.ArgumentParser(description="Run multiple training series for a dataset across backbones and modes")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., MUTAG, PROTEINS, ogbg-molhiv)')
    parser.add_argument('--device', type=int, default=0, help='GPU device index (default: 0)')
    parser.add_argument('--epochs', type=int, default=500, help='Epochs per run (default: 500)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--early_stop', type=int, default=7)
    parser.add_argument('--runs', type=int, default=10, help='Number of seed runs (default: 10)')
    parser.add_argument('--backbones', type=str, nargs='*', default=['GAT', 'GCN', 'GIN', 'GraphSAGE'],
                        help='Backbones to run')
    parser.add_argument('--modes', type=str, nargs='*', default=['P', 'T', 'S'], help='Train modes to run')
    parser.add_argument('--python', type=str, default=sys.executable, help='Python executable to use')
    parser.add_argument('--extra', type=str, nargs=argparse.REMAINDER, default=[],
                        help='Any extra args to pass to main.py (put them after --)')

    args = parser.parse_args()

    # Generate seeds: one seed per run, applied to all backbones and modes within that run
    rng = random.SystemRandom()
    seeds = [rng.randint(1, 2**31 - 1) for _ in range(args.runs)]

    print(f"Running {args.runs} runs for dataset={args.dataset}")
    print("Backbones:", args.backbones)
    print("Modes:", args.modes)
    print("Seeds per run:")
    for i, s in enumerate(seeds, 1):
        print(f"  Run {i}: seed={s}")

    for run_idx, seed in enumerate(seeds, 1):
        print("\n" + "=" * 80)
        print(f"Run {run_idx}/{args.runs} — shared seed={seed} for all configurations")
        print("=" * 80)
        for backbone in args.backbones:
            for mode in args.modes:
                cmd = [
                    args.python, 'main.py',
                    '--dataset', args.dataset,
                    '--device', str(args.device),
                    '--epochs', str(args.epochs),
                    '--batch_size', str(args.batch_size),
                    '--nhid', str(args.nhid),
                    '--nlayers', str(args.nlayers),
                    '--gat_heads', str(args.gat_heads),
                    '--dropout', str(args.dropout),
                    '--weight_decay', str(args.weight_decay),
                    '--alpha', str(args.alpha),
                    '--tau', str(args.tau),
                    '--early_stop', str(args.early_stop),
                    '--train_mode', mode,
                    '--backbone', backbone,
                    '--seed', str(seed),
                    '--runs', '1'  # ensure single training per configuration
                ]
                if args.extra:
                    cmd.extend(args.extra)

                stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{stamp}] Starting: dataset={args.dataset}, backbone={backbone}, mode={mode}, seed={seed}")
                print("Command:", " ".join(cmd))
                ret = subprocess.call(cmd)
                if ret != 0:
                    print(f"[WARN] Command failed with exit code {ret}")


if __name__ == '__main__':
    main()
