import argparse
import json
import os
from function_circuit.function_utils import set_global_seed
import pandas as pd

def sample_csv(input_csv_path, num_train_sample, num_test_seq, seed, single_only=False):
    num_sample_per_bin = int(num_train_sample / 2)
    df = pd.read_csv(input_csv_path)
    
    if single_only:
        df = df[~df['mutant'].str.contains(':', na=False)]
    
    better = df[df['DMS_score_bin'] == 1]
    worse = df[df['DMS_score_bin'] == 0]

    sampled_better = better.sample(n=num_sample_per_bin, random_state=seed)
    sampled_worse = worse.sample(n=num_sample_per_bin, random_state=seed)
    
    train_df = pd.concat([sampled_better, sampled_worse])
    test_df = df.drop(train_df.index)

    better_test = test_df[test_df['DMS_score_bin'] == 1]
    worse_test = test_df[test_df['DMS_score_bin'] == 0]

    sampled_better_test = better_test.sample(n=num_test_seq//2, random_state=seed)
    sampled_worse_test = worse_test.sample(n=num_test_seq//2, random_state=seed)
    sampled_test_df = pd.concat([sampled_better_test, sampled_worse_test])

    train_df = train_df.reset_index(drop=True)
    sampled_test_df = sampled_test_df.reset_index(drop=True)
    
    train_sequences = get_sequence_objects(train_df)
    test_sequences = get_sequence_objects(sampled_test_df)

    return train_sequences, test_sequences

def get_sequence_objects(df, output_json_path=None):
    sequences = []
    for row in df.itertuples():
        sequences.append(
            {
                "mutated_sequence": row.mutated_sequence,
                "DMS_score": row.DMS_score,
                "DMS_score_bin": row.DMS_score_bin,
                "mutations": row.mutant.split(":")
            }
        )
    result = {"sequences": sequences}

    if output_json_path is not None:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, sort_keys=True)
    else:
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+', required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_sample", type=int, default=128)
    args = parser.parse_args()

    output_dir = "function_circuit/sampled_mutations"
    os.makedirs(output_dir, exist_ok=True)

    set_global_seed(args.seed)
    
    for dataset in args.datasets:
        base_name = os.path.basename(dataset)
        new_filename = base_name.replace(".csv", "_mutations.json")
        
        output_json = os.path.join(output_dir, new_filename)
        
        print(f"Processing: {base_name} -> Saving to: {output_json}")
        
        train_df, test_df = sample_csv(dataset, args.num_sample, args.seed, single_only=False)
        get_sequence_objects(train_df, output_json)