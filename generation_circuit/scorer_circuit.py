import torch
import torch.nn as nn
from typing import Any, List

from progen3.scorer import ProGen3Scorer
from generation_circuit.generation_utils import TaskSpec, token_is_amino_acid


class ScorerCircuit(ProGen3Scorer):
    def __init__(self, model, tokenizer, max_batch_tokens: int = 65536, reduction: str = "mean"):
        super().__init__(model, max_batch_tokens, reduction)
        self.tokenizer = tokenizer

    def score_batch_with_tasks(self, sequences: List[str], tasks: List[TaskSpec]) -> dict[str, List[float]]:
        """
        Score sequences, but only on the generated parts based on tasks.
        Only scores forward direction, no reverse.
        """
        kwargs_n_to_c = self.batch_preparer.get_batch_kwargs(sequences, device=self.model.device, reverse=False)
        
        generated_masks = []
        for i, (seq, task) in enumerate(zip(sequences, tasks)):
            input_ids = kwargs_n_to_c["input_ids"][i]
            labels = kwargs_n_to_c["labels"][i]
            
            # Compute aa_positions
            aa_positions = torch.tensor([j for j in range(len(input_ids)) if token_is_amino_acid(self.tokenizer.id_to_token(int(input_ids[j])))], dtype=torch.long)
            
            mask = torch.zeros_like(labels, dtype=torch.bool)
            
            if task.task_type == "CLM":
                start_res = len(task.prompt)
            else:
                start_res = int(task.span_start or 0)
            
            start_res = max(0, min(start_res, int(aa_positions.numel())))
            end_res = max(start_res, min(start_res + len(task.target_sequence), int(aa_positions.numel())))
            
            selected_token_positions = aa_positions[start_res:end_res]
            if selected_token_positions.numel() > 0:
                mask[selected_token_positions] = True
            
            # generated_mask for targets (shifted)
            generated_mask = mask[1:]
            generated_masks.append(generated_mask)
        
        output_batch = self._log_likelihoods(kwargs_n_to_c, generated_masks)
        
        scores: dict[str, List[float]] = {"log_likelihood": [], "perplexity": []}
        for i in range(len(sequences)):
            ll = output_batch[i]
            scores["log_likelihood"].append(ll.item())
            scores["perplexity"].append(torch.exp(-ll).item())
        return scores

    def _log_likelihoods(self, model_forward_kwargs: dict[str, Any], generated_masks: List[torch.Tensor]) -> torch.Tensor:
        output = self.model(
            input_ids=model_forward_kwargs["input_ids"],
            labels=model_forward_kwargs["labels"],
            sequence_ids=model_forward_kwargs["sequence_ids"],
            position_ids=model_forward_kwargs["position_ids"],
            return_dict=True,
        )
        labels = model_forward_kwargs["labels"]
        target_mask = labels != self.model.config.pad_token_id

        targets = labels[..., 1:].contiguous()
        target_mask = target_mask[..., 1:].contiguous()
        
        # Apply generated mask
        batch_generated_mask = torch.stack(generated_masks, dim=0)
        target_mask = target_mask & batch_generated_mask
        
        logits = output.logits[..., :-1, :].contiguous().to(torch.float32)
        flat_logits = logits.view(-1, logits.shape[-1])
        nll = nn.functional.cross_entropy(flat_logits, targets.view(-1), reduction="none").view(targets.shape)
        nll = (nll * target_mask.to(nll)).sum(dim=1)
        if self.reduction == "mean":
            nll = nll / target_mask.sum(dim=1)
        return -nll.detach()