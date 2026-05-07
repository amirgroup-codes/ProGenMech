import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

class CLTPLTReconstructionScorer:
    def __init__(self, discoverer, reduction="mean"):
        self.discoverer = discoverer
        self.reduction = reduction

    def _get_log_likelihoods(self, sequences, active_nodes=None, sequential=True, freeze_attention=True, reverse=False, **clt_kwargs):
        # 1. Handle string reversal for bidirectional scoring
        if reverse:
            # Reversing strings for C-to-N pass
            proc_seqs = [s[::-1] for s in sequences]
        else:
            proc_seqs = sequences

        # 2. Get Reconstructed Logits (Float32)
        logits = self.discoverer.reconstruct_logits(batch_seqs=proc_seqs, active_nodes=active_nodes, sequential=sequential, freeze_attention=freeze_attention, **clt_kwargs)

        # 3. Get labels and padding mask
        model_inputs = self.discoverer._prepare_inputs(proc_seqs)
        labels = model_inputs["labels"].to(logits.device)
        pad_id = self.discoverer.progen.config.pad_token_id
        
        # 4. Standard Causal Shift
        # Logits at index t predict label at t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        del logits  # Free GPU memory immediately
        
        # 5. Compute NLL
        nll_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        ).view(shift_labels.size())
        del shift_logits  # Free GPU memory

        # Mask padding and reduction
        mask = (shift_labels != pad_id).float()
        nll = (nll_loss * mask).sum(dim=1)
        
        if self.reduction == "mean":
            nll = nll / mask.sum(dim=1)
        
        del nll_loss, mask, shift_labels  # Free GPU memory
            
        return -nll.cpu().detach()  # Move to CPU to free GPU memory

    @torch.no_grad()
    def score(self, sequences, task, active_nodes=None, **clt_kwargs):
        ll_fwd = self._get_log_likelihoods(sequences, active_nodes=active_nodes, sequential=task['sequential'], freeze_attention=task['freeze_attention'], reverse=False, **clt_kwargs)
        torch.cuda.empty_cache()  # Free GPU memory between forward and reverse
        ll_rev = self._get_log_likelihoods(sequences, active_nodes=active_nodes, sequential=task['sequential'], freeze_attention=task['freeze_attention'], reverse=True, **clt_kwargs)
        torch.cuda.empty_cache()  # Free GPU memory after reverse
        
        avg_ll = (ll_fwd + ll_rev) / 2
        return {"log_likelihood": avg_ll.tolist()}