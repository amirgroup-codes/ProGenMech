import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from scipy import stats

import numpy as np
import pandas as pd
import torch
import re


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def token_is_amino_acid(token: str) -> bool:
    """
    Return True when a tokenizer token is a single amino-acid residue.
    """
    return len(token) == 1 and token in AA_ALPHABET

def token_ids_to_aa_mask_and_tokens(tokenizer, token_ids: Sequence[int]) -> Tuple[List[bool], List[str]]:
    """
    Convert token ids into amino-acid masks and tokenizer strings.
    """
    aa_mask = []
    tok_texts = []
    for tid in token_ids:
        tok = tokenizer.id_to_token(int(tid))
        tok_texts.append(tok)
        aa_mask.append(token_is_amino_acid(tok))
    return aa_mask, tok_texts

def parse_mutation(mutation_str):
    """
    Parses a mutation string (e.g., 'A10L') into its components.
    
    Returns:
        tuple: (original, position, mutated)
    """
    # Regex to capture the initial letter(s), the middle digits, and trailing letter(s)
    match = re.match(r"([A-Z]+)([0-9]+)([A-Z]+)", mutation_str, re.I)
    
    if match:
        original = match.group(1)
        position = int(match.group(2))
        mutated = match.group(3)
        
        return {
            "original": original,
            "position": position,
            "mutated": mutated,
            "mapping": f"{original}->{mutated}"
        }
    else:
        raise ValueError(f"Invalid mutation format: {mutation_str}")

def get_spearman_p(scores, experimental_values):
    """
    Compute Spearman correlation and p-value between scores and experimental values.
    """
    if len(scores) == 0 or len(experimental_values) == 0:
        return 0.0, 1.0
    corr, p_value = stats.spearmanr(scores, experimental_values)
    return corr, p_value
