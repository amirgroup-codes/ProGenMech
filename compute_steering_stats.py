import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "steering_results"
all_dataset_paths = {}
for dataset_dir in sorted(ROOT.iterdir()):
    if not dataset_dir.is_dir() or dataset_dir.name == '__pycache__':
        continue
    csv_paths = []
    for fold_dir in sorted(dataset_dir.glob('fold*')):
        if not fold_dir.is_dir():
            continue
        for supp_dir in sorted(fold_dir.glob('supp*')):
            if not supp_dir.is_dir():
                continue
            for csv_path in sorted(supp_dir.glob('*.csv')):
                csv_paths.append(csv_path)
    if csv_paths:
        all_dataset_paths[dataset_dir.name] = csv_paths

if not all_dataset_paths:
    raise SystemExit('No CSV files found under steering_results/{dataset}/fold*/supp*/*.')

methods = ['wildtype', 'progen3', 'clt', 'plt']
stats_metrics = ['mean', 'max', 'top10', 'top20']
win_counts = {
    metric: {'clt': 0, 'plt': 0, 'tie': 0}
    for metric in stats_metrics
}

for dataset_name, csv_paths in sorted(all_dataset_paths.items()):
    frames = []
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f'Warning: failed to read {csv_path}: {exc}')
            continue
        frames.append(df)

    if not frames:
        print(f'No valid CSV data could be read for dataset {dataset_name}.')
        continue

    big_df = pd.concat(frames, ignore_index=True)
    big_df = big_df.drop_duplicates()

    if 'method' not in big_df.columns or 'log_likelihood' not in big_df.columns:
        print(f'Required columns missing for dataset {dataset_name}: method and/or log_likelihood.')
        continue

    big_df['log_likelihood'] = pd.to_numeric(big_df['log_likelihood'], errors='coerce')

    print(f'Dataset: {dataset_name}')
    print(f'  files loaded: {len(csv_paths)}')
    print(f'  rows after concatenation: {len(big_df)}')
    print(f'  rows after deduplication: {len(big_df.drop_duplicates())}')

    method_stats = {}
    for method in methods:
        subset = big_df[big_df['method'] == method].copy()
        subset = subset.loc[subset['log_likelihood'].notna()]
        if subset.empty:
            print(f'  {method}: no rows found')
            continue

        n = len(subset)
        top10_n = max(1, int(round(n * 0.10)))
        top20_n = max(1, int(round(n * 0.20)))
        mean_ll = subset['log_likelihood'].mean()
        std_ll = subset['log_likelihood'].std(ddof=0)
        max_ll = subset['log_likelihood'].max()
        top10 = subset['log_likelihood'].nlargest(top10_n)
        top20 = subset['log_likelihood'].nlargest(top20_n)
        top10_mean_ll = top10.mean()
        top20_mean_ll = top20.mean()
        top10_std_ll = top10.std(ddof=0)
        top20_std_ll = top20.std(ddof=0)

        method_stats[method] = {
            'mean': mean_ll,
            'std': std_ll,
            'max': max_ll,
            'top10': top10_mean_ll,
            'top10_std': top10_std_ll,
            'top20': top20_mean_ll,
            'top20_std': top20_std_ll,
            'n': n,
        }

        print(
            f'  {method}: mean={mean_ll:.6f} +/- {std_ll:.6f}, max={max_ll:.6f}, '
            f'top10_mean={top10_mean_ll:.6f} +/- {top10_std_ll:.6f}, '
            f'top20_mean={top20_mean_ll:.6f} +/- {top20_std_ll:.6f}, n={n}'
        )

    if 'clt' in method_stats and 'plt' in method_stats:
        for metric in stats_metrics:
            clt_value = method_stats['clt'][metric]
            plt_value = method_stats['plt'][metric]
            if clt_value > plt_value:
                win_counts[metric]['clt'] += 1
            elif plt_value > clt_value:
                win_counts[metric]['plt'] += 1
            else:
                win_counts[metric]['tie'] += 1

    print('')

print('CLT vs PLT win counts across all datasets:')
for metric in stats_metrics:
    counts = win_counts[metric]
    print(
        f'  {metric}: CLT={counts["clt"]}, PLT={counts["plt"]}, ties={counts["tie"]}'
    )
