from tqdm import tqdm
import numpy as np

def compute_attribution(
    discoverer,
    seqs,
    batch_size=8,
    sequential=False,
    freeze_attention=True,
    source="layer_output",
    generated_mask_fn=None,
):
    global_attr = {}
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        model_inputs = discoverer._prepare_inputs(batch)
        if generated_mask_fn is None:
            generated_mask_BT = model_inputs["labels"] != discoverer.pad_id
        else:
            generated_mask_BT = generated_mask_fn(batch, model_inputs)
        print(batch)
        attr_batch = discoverer.get_gradients(
            batch,
            sequential=sequential,
            freeze_attention=freeze_attention,
            source=source,
            generated_mask_BT=generated_mask_BT,
        )
        for l, scores in attr_batch.items():
            if l not in global_attr:
                global_attr[l] = np.zeros_like(scores)
            global_attr[l] += scores 
     
    return global_attr

def rank_nodes(global_attr):
    """
    Flattens and ranks nodes by attribution score.
    Returns: List of (layer, node_idx, score) sorted descending.
    """
    ranking = []
    for l, scores in global_attr.items():
        for idx, s in enumerate(scores):
            ranking.append((l, idx, s))
    ranking.sort(key=lambda x: x[2], reverse=True)
    return ranking

def circuit_search(
    discoverer, 
    probe, 
    ranking, 
    val_seqs, 
    val_y, 
    target_metric, 
    metric_fn, # e.g. evaluate_circuit for F1, or evaluate_regression for Pearson
    batch_size=8,
    step_size=32,
    max_nodes=1000,
    desc="Scanning",
    metric_name="score",
    display_transform=None,
    **kwargs
):
    """
    Performs selection of circuit nodes.
    Returns: (best_nodes_dict, best_k, best_metric_val)
    """
    if display_transform is None:
        display_transform = lambda x: x

    best_nodes = {}
    max_nodes = min(max_nodes, len(ranking))
    best_k = max_nodes
    best_metric_val = -float('inf') 
    highest_seen_metric = -float('inf')
    best_seen_config = None # Will store (nodes, k, metric)
    step_values = list(range(step_size, max_nodes + 1, step_size))
    if max_nodes > 0 and (not step_values or step_values[-1] != max_nodes):
        step_values.append(max_nodes)
    step_iter = step_values
    with tqdm(step_iter, desc=desc, position=1, leave=False) as pbar:
        for k in pbar:
            # 1. Select top k nodes
            top_k = ranking[:k]
            active = {}
            for l, n, _ in top_k:
                if l not in active: active[l] = set()
                active[l].add(n)

            # 2. Evaluate
            # metric_fn must accept (discoverer, probe, seqs, y, active_nodes, batch_size)
            curr_metric = metric_fn(discoverer, probe, val_seqs, val_y, active, batch_size)
            if curr_metric > highest_seen_metric:
                highest_seen_metric = curr_metric
                best_seen_config = (active, k, curr_metric)

            shown_curr = display_transform(curr_metric)
            shown_target = display_transform(target_metric)
            shown_peak = display_transform(highest_seen_metric)
            pbar.set_postfix({
                "nodes": k,
                metric_name: f"{shown_curr:.3f}",
                "target": f"{shown_target:.3f}",
                "peak": f"{shown_peak:.3f}"
            })
            
            # 3. Check stopping condition
            if curr_metric >= target_metric:
                best_nodes = active
                best_k = k
                best_metric_val = curr_metric
                break
    
    # If we never reached the target, take the best config
    if not best_nodes:
        if best_seen_config is not None:
            best_nodes, best_k, best_metric_val = best_seen_config
        else:
            best_k = max_nodes
            top_k = ranking[:best_k]
            for l, n, _ in top_k:
                if l not in best_nodes: best_nodes[l] = set()
                best_nodes[l].add(n)
            best_metric_val = metric_fn(discoverer, probe, val_seqs, val_y, best_nodes, batch_size, **kwargs)
            
    return best_nodes, best_k, best_metric_val
