import argparse
import csv
import json
import os
import sys
from pathlib import Path
import time
import gc

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from function_circuit.function_utils import (
    get_spearman_p,
    set_global_seed,
)
from circuit_utils.clt_circuit import CircuitDiscovererCLT
from function_circuit.clt_plt_scorer import CLTPLTReconstructionScorer

from function_circuit.prepare_data import sample_csv

from circuit_utils.plt_circuit import CircuitDiscovererPLT
from external.progen3.src.progen3.scorer import ProGen3Scorer

# datasets = ['../data/DMS_ProteinGym_substitutions/A4_HUMAN_Seuma_2022.csv',
#             '../data/DMS_ProteinGym_substitutions/AMFR_HUMAN_Tsuboyama_2023_4G3O.csv',
#             '../data/DMS_ProteinGym_substitutions/BBC1_YEAST_Tsuboyama_2023_1TG0.csv',
#             '../data/DMS_ProteinGym_substitutions/CAPSD_AAV2S_Sinai_2021.csv',
#             '../data/DMS_ProteinGym_substitutions/DLG4_HUMAN_Faure_2021.csv',
#             '../data/DMS_ProteinGym_substitutions/F7YBW8_MESOW_Ding_2023.csv',
#             '../data/DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv',
#             '../data/DMS_ProteinGym_substitutions/GRB2_HUMAN_Faure_2021.csv',
#             '../data/DMS_ProteinGym_substitutions/HIS7_YEAST_Pokusaeva_2019.csv',
#             '../data/DMS_ProteinGym_substitutions/RASK_HUMAN_Weng_2022_abundance.csv',
#             '../data/DMS_ProteinGym_substitutions/SPG1_STRSG_Olson_2014.csv',
#             '../data/DMS_ProteinGym_substitutions/YAP1_HUMAN_Araya_2012.csv',
#             '../data/DMS_ProteinGym_substitutions/PTEN_HUMAN_Mighell_2018.csv',
#             '../data/DMS_ProteinGym_substitutions/BLAT_ECOLX_Stiffler_2015.csv',
#             '../data/DMS_ProteinGym_substitutions/P53_HUMAN_Kotler_2018.csv']

METHOD_CONFIGS = {
    "clt_direct": {
        "discoverer_key": "clt",
        "sequential": False,
        "freeze_attention": True,
    },
    "clt_sequential_freeze": {
        "discoverer_key": "clt",
        "sequential": True,
        "freeze_attention": True,
    },
    "clt_sequential_unfreeze": {
        "discoverer_key": "clt",
        "sequential": True,
        "freeze_attention": False,
    },
    "plt_sequential_freeze": {
        "discoverer_key": "plt",
        "sequential": True,
        "freeze_attention": True,
    },
    "plt_sequential_unfreeze": {
        "discoverer_key": "plt",
        "sequential": True,
        "freeze_attention": False,
    },
}

def compute_baseline_scores(
    progen_scorer : ProGen3Scorer,
    clt_scorer : CLTPLTReconstructionScorer,
    plt_scorer : CLTPLTReconstructionScorer,
    sequences: list[str],
    scores: list[float],
    batch_size: int = 8
) -> dict:
    
    result_progen = progen_scorer.evaluate(sequences)['log_likelihood']
    spearman_progen, _ = get_spearman_p(result_progen, scores)
    
    clt_plt_results = {"progen": spearman_progen}
    
    for task_name, task_config in METHOD_CONFIGS.items():
        result = []
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Scoring {task_name}"):
            if task_config['discoverer_key'] == 'clt':
                batch_result = clt_scorer.score(sequences[i:i+batch_size], task=task_config)['log_likelihood']
                clt_scorer.discoverer.clear_cache()
            elif task_config['discoverer_key'] == 'plt':
                batch_result = plt_scorer.score(sequences[i:i+batch_size], task=task_config)['log_likelihood']
                plt_scorer.discoverer.clear_cache()
            result.extend(batch_result)
        
        spearman_task, _ = get_spearman_p(result, scores)
        
        clt_plt_results[task_name] = spearman_task

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    return clt_plt_results

def discover_circuits(
    circuit_discoverer,
    model_scorer : CLTPLTReconstructionScorer,
    sequences: list[str],
    dataset_scores: list[float],
    progen_base: float,
    task_base: float,
    task_name: str,
    task_config : dict,
    batch_size: int = 8,
    num_train_seq: int = 256,
):

    # Circuit discovery algorithm, following implementation in ProtoMech
    theta = min(0.7 * progen_base, task_base)

    all_gradients = {}

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        batch_grads = circuit_discoverer.get_gradients(batch, sequential=task_config['sequential'], freeze_attention=task_config['freeze_attention'], zero_shot=True)
        
        for layer, score in batch_grads.items():
            if layer not in all_gradients:
                all_gradients[layer] = score
            else:
                all_gradients[layer] += score
                
        torch.cuda.empty_cache()

    top_1000 = rank_latents(all_gradients)

    latents = 32
    score = float('-inf')

    while latents <= 1000 and score < theta:
        current_pool = top_1000[:latents]
        active_nodes = {}
        for node in current_pool:
            active_nodes.setdefault(node['layer'], []).append(int(node['idx']))
        
        start_time = time.time()
        circuit_log_likelihoods = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            batch_result = model_scorer.score(batch, task=task_config, active_nodes=active_nodes)['log_likelihood']
            circuit_log_likelihoods.extend(batch_result)
        spearman_circuit, _ = get_spearman_p(circuit_log_likelihoods, dataset_scores)
        duration = time.time() - start_time
        score = spearman_circuit

        print(f"Nodes: {latents} | Spearman: {spearman_circuit:.4f} | Time: {duration:.1f}s")

        if latents == 1000:
            print("Reached max latents with score:", score)
            break
        if score >= theta:
            print(f"Threshold reached with {latents} latents!")
            break

        # Cut off at 1000 if step exceeds 1000 latents, otherwise increment by 32
        latents += 32
        if latents > 1000:
            latents = 1000

    final_data = {
        "method": task_name,
        "n_train": num_train_seq,
        "k": latents,
        "sequential": task_config['sequential'],
        "freeze_attention": task_config['freeze_attention'],
        "nodes": active_nodes
    }

    return final_data
            

def rank_latents(latent_scores: dict) -> dict:
    """
    Ranks latents based on their scores and returns the top 1000.
    """
    all_latents = []
    for layer, scores in latent_scores.items():
        for idx, s in enumerate(scores):
            all_latents.append({'layer': layer, 'idx': idx, 'score': s})

    all_latents.sort(key=lambda x: x['score'], reverse=True)
    top_1000_pool = all_latents[:1000]
    
    return top_1000_pool

def circuit_discovery_main(
    circuit_discoverer_clt : CircuitDiscovererCLT,
    circuit_discoverer_plt : CircuitDiscovererPLT,
    clt_scorer : CLTPLTReconstructionScorer,
    plt_scorer : CLTPLTReconstructionScorer,
    progen_scorer : ProGen3Scorer,
    dataset_csv: str,
    batch_size: int = 8,
    num_train_seq: int = 256,
    num_test_seq: int = 1000
):
    """
    Main function for circuit discovery on a single DMS assay
    """
    for fold_idx in range(5):
        seed = 42 + fold_idx
        set_global_seed(seed)
        print(f"Running circuit discovery with seed {seed}...")
        train_json, test_json = sample_csv(dataset_csv, num_train_sample=num_train_seq, num_test_seq=num_test_seq, seed=seed, single_only=False)
        sequences = [s['mutated_sequence'] for s in train_json['sequences']]
        dataset_scores = [s['DMS_score'] for s in train_json['sequences']]
        test_sequences = [s['mutated_sequence'] for s in test_json['sequences']]
        test_dataset_scores = [s['DMS_score'] for s in test_json['sequences']]

        # Obtain scores for each method on the training samples
        base_scores = compute_baseline_scores(progen_scorer, clt_scorer, plt_scorer, sequences, dataset_scores, batch_size)
        # Obtain clean spearman and max spearman values for the test set
        base_test_scores = compute_baseline_scores(progen_scorer, clt_scorer, plt_scorer, test_sequences, test_dataset_scores, batch_size)
        progen_base = base_scores['progen']
        for task_name, task_config in METHOD_CONFIGS.items():
            task_base = base_scores[task_name]
            if task_config['discoverer_key'] == 'clt':
                circuit_data = discover_circuits(circuit_discoverer_clt, clt_scorer, sequences, dataset_scores, progen_base, task_base, task_name, task_config, batch_size, num_train_seq)
                curr_scorer = clt_scorer
            else:
                circuit_data = discover_circuits(circuit_discoverer_plt, plt_scorer, sequences, dataset_scores, progen_base, task_base, task_name, task_config, batch_size, num_train_seq)
                curr_scorer = plt_scorer
            circuit_data['clean_spearman'] = base_test_scores['progen']
            circuit_data['max_spearman'] = base_test_scores[task_name]

            # Compute spearman (0 latents active) on test data
            circuit_log_likelihoods = []
            for i in tqdm(range(0, len(test_sequences), batch_size), desc="No Latents Test Pass"):
                batch = test_sequences[i : i + batch_size]
                batch_result = curr_scorer.score(batch, task=task_config, active_nodes={})['log_likelihood']
                circuit_log_likelihoods.extend(batch_result)
            
            spearman_no_latents, _ = get_spearman_p(circuit_log_likelihoods, test_dataset_scores)
            circuit_data['no_latents_spearman'] = spearman_no_latents

            # Compute spearman (circuit) on test data
            circuit_log_likelihoods = []
            for i in tqdm(range(0, len(test_sequences), batch_size), desc="Circuit Test Pass"):
                batch = test_sequences[i : i + batch_size]
                batch_result = curr_scorer.score(batch, task=task_config, active_nodes=circuit_data['nodes'])['log_likelihood']
                circuit_log_likelihoods.extend(batch_result)
            
            spearman_circuit, _ = get_spearman_p(circuit_log_likelihoods, test_dataset_scores)
            circuit_data['recovered_spearman'] = spearman_circuit

            save_dir = Path(f"function_circuit/circuits/{task_name}/{Path(dataset_csv).stem}")
            save_dir.mkdir(parents=True, exist_ok=True)

            circuit_data_json = save_dir / f"seq{num_train_seq}_fold{fold_idx}.json"
            with open(circuit_data_json, 'w') as f:
                json.dump(circuit_data, f, indent=4)

            del circuit_log_likelihoods
            circuit_discoverer_clt.clear_cache()
            circuit_discoverer_plt.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        del base_scores, base_test_scores, sequences, dataset_scores, test_sequences, test_dataset_scores
        circuit_discoverer_clt.clear_cache()
        circuit_discoverer_plt.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def dataset_scores(circuit_discoverer : CircuitDiscovererCLT, output_csv: str, disable_tqdm: bool = False, batch_size: int = 8) -> None:
    clt_scorer = CLTPLTReconstructionScorer(discoverer=circuit_discoverer)
    progen_scorer = ProGen3Scorer(circuit_discoverer.progen)
    
    # Check if file exists to determine if we need to write headers
    file_exists = Path(output_csv).exists()
    
    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['dataset', 'p_progen', 'p_clt_direct', 'p_clt_seq_freeze', 'p_clt_seq_unfreeze']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for dataset in datasets:
            with open(dataset, 'r') as file:
                df = pd.read_csv(file)

            sample_n = min(1000, len(df))
            sampled = df.sample(n=sample_n, random_state=42)

            # Results for single mutations with ProGen3 112m
            experimental_scores = sampled['DMS_score'].tolist()
            result_progen = progen_scorer.evaluate(sampled['mutated_sequence'].tolist())['log_likelihood']
            p_progen, _ = get_spearman_p(result_progen, experimental_scores)

            seqs = sampled['mutated_sequence'].tolist()
            result_clt_direct = []
            result_clt_seq_freeze = []
            result_clt_seq_unfreeze = []
            for i in tqdm(range(0, len(seqs), batch_size), desc="Scoring mutations", disable=disable_tqdm):
                batch_result_clt_direct = clt_scorer.score(seqs[i:i+batch_size], task=METHOD_CONFIGS['clt_direct'])['log_likelihood']
                result_clt_direct.extend(batch_result_clt_direct)
                batch_result_clt_seq_freeze = clt_scorer.score(seqs[i:i+batch_size], task=METHOD_CONFIGS['clt_sequential_freeze'])['log_likelihood']
                result_clt_seq_freeze.extend(batch_result_clt_seq_freeze)
                batch_result_clt_seq_unfreeze = clt_scorer.score(seqs[i:i+batch_size], task=METHOD_CONFIGS['clt_sequential_unfreeze'])['log_likelihood']
                result_clt_seq_unfreeze.extend(batch_result_clt_seq_unfreeze)
        
            p_clt_direct,  _ = get_spearman_p(result_clt_direct, experimental_scores)
            p_clt_seq_freeze,  _ = get_spearman_p(result_clt_seq_freeze, experimental_scores)
            p_clt_seq_unfreeze,  _ = get_spearman_p(result_clt_seq_unfreeze, experimental_scores)

            # Extract dataset name from path
            dataset_name = Path(dataset).stem
            
            # Write results to CSV
            writer.writerow({
                'dataset': dataset_name,
                'p_progen': p_progen,
                'p_clt_direct': p_clt_direct,
                'p_clt_seq_freeze': p_clt_seq_freeze,
                'p_clt_seq_unfreeze': p_clt_seq_unfreeze,
            })
            print(f"{dataset_name}: {p_progen} {p_clt_direct} {p_clt_seq_freeze} {p_clt_seq_unfreeze}")
            del df, sampled, result_clt_direct, result_clt_seq_freeze, result_clt_seq_unfreeze
            gc.collect() 
            torch.cuda.empty_cache()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+', required=True)
    parser.add_argument("--torch_num_threads", type=int, default=4)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for scoring (reduce if OOM)")
    parser.add_argument("--num_train_seq", type=int, default=256, help="Number of sequences to use for circuit discovery")
    parser.add_argument("--num_test_seq", type=int, default=1000, help="Number of sequences to evaluate on for test set")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    clt_ckpt = os.environ.get("CLT_CHECKPOINT")
    plt_ckpt = os.environ.get("PLT_CHECKPOINT")
    if not clt_ckpt or not plt_ckpt:
        raise ValueError("CLT_CHECKPOINT and PLT_CHECKPOINT must be set in environment")

    print(f"[main] device: {device}")
    print("[main] Loading discoverers...")
    clt_discoverer = CircuitDiscovererCLT(device=device, ckpt_path=clt_ckpt)
    plt_discoverer = CircuitDiscovererPLT(device=device, ckpt_path=plt_ckpt)
    clt_scorer = CLTPLTReconstructionScorer(discoverer=clt_discoverer)
    plt_scorer = CLTPLTReconstructionScorer(discoverer=plt_discoverer)
    progen_scorer = ProGen3Scorer(clt_discoverer.progen)

    for dataset_csv in args.datasets:
        print(f"Processing DMS: {dataset_csv}")
        circuit_discovery_main(clt_discoverer, plt_discoverer, clt_scorer, plt_scorer, progen_scorer, dataset_csv, batch_size=args.batch_size, num_train_seq=args.num_train_seq, num_test_seq=args.num_test_seq)
    

if __name__ == '__main__':
    main()