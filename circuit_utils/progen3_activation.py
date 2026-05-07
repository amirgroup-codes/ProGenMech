import torch


# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size
# L: Total number of LM layers
# T: Sequence length of protein (variable)
# D: PLT Latent dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# S: B * T
# ──────────────────────────────────────────────────────────────────────────────


class ProGen3ActivationCollector:
	"""
	Grabs MLP inputs, outputs, and layer embeddings.
	More complicated than the version in training/clt_module.py
	"""
	def __init__(self, progen3_model, vocabulary, target_layers=None):
		self.model = progen3_model
		self.vocabulary = vocabulary
		self.target_layers = target_layers if target_layers else list(range(len(progen3_model.model.layers)))
		self.activations = {}
		self.hooks = []
		self.cache = {}  # Storage for pre-computed activations

	def clear_cache(self):
		"""Removes cached tensors to free up GPU memory."""
		self.cache.clear()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def _make_hook(self, key, scale=1.0, transpose=False, capture_input=False, tuple_index=0):
		"""
		Creates a hook that:
		1. Extracts tensor from tuple (if needed)
		2. Scales the tensor (for embeddings)
		3. Transposes (T, B, H) -> (B, T, H) (for layers)
		"""

		def hook(module, input, output):
			if capture_input:
				data = input[tuple_index] if isinstance(input, tuple) else input
			else:
				if isinstance(output, tuple):
					data = output[tuple_index]
				else:
					data = output
			if scale != 1.0:
				data = data * scale
			# (T, B, H) -> (B, T, H)
			if transpose:
				data = data.transpose(0, 1)
			self.activations[key] = data.detach().to("cpu", dtype=torch.float32, non_blocking=True)
		return hook

	def register_hooks(self):
		self.remove_hooks()
		# 1. Hook Embeddings (B, T, H)
		# ProGen3 embeddings include token + sequence embeddings before layer 0.
		first_layer = self.model.model.layers[0]
		embed_hook = first_layer.register_forward_hook(
			self._make_hook(-1, scale=1.0, transpose=False, capture_input=True, tuple_index=0)
		)
		self.hooks.append(embed_hook)

		# 2. Hook Layers (B, T, H)
		for layer_idx in self.target_layers:
			layer_module = self.model.model.layers[layer_idx]
			hook = layer_module.register_forward_hook(
				self._make_hook(layer_idx, scale=1.0, transpose=False, capture_input=False, tuple_index=0)
			)
			self.hooks.append(hook)

		# 3. Hook MLP Inputs (residual stream before and after post_attention_layernorm)
		for layer_idx in self.target_layers:
			layer = self.model.model.layers[layer_idx]
			if hasattr(layer, "post_attention_layernorm"):
				mlp_norm = layer.post_attention_layernorm
				# 3a. Hook the INPUT to post_attention_layernorm (residual stream before normalization)
				hook = mlp_norm.register_forward_hook(
					self._make_hook(f"mlp_input_{layer_idx}", scale=1.0, transpose=False, capture_input=True)
				)
				self.hooks.append(hook)
				# 3b. Hook the OUTPUT of post_attention_layernorm (post-LN residual stream used for CLT inference)
				hook = mlp_norm.register_forward_hook(
					self._make_hook(f"clt_input_{layer_idx}", scale=1.0, transpose=False, capture_input=False)
				)
				self.hooks.append(hook)
			else:
				# Fused attention+norm path: norm_attn_norm returns
				# (post_attention_layernorm_output, residual_before_post_attention_layernorm, ...)
				norm_attn_norm = layer.norm_attn_norm
				hook = norm_attn_norm.register_forward_hook(
					self._make_hook(
						f"mlp_input_{layer_idx}",
						scale=1.0,
						transpose=False,
						capture_input=False,
						tuple_index=1,
					)
				)
				self.hooks.append(hook)
				hook = norm_attn_norm.register_forward_hook(
					self._make_hook(
						f"clt_input_{layer_idx}",
						scale=1.0,
						transpose=False,
						capture_input=False,
						tuple_index=0,
					)
				)
				self.hooks.append(hook)

		# 4. Hook MLP Outputs (block_sparse_moe output, before residual add)
		for layer_idx in self.target_layers:
			moe = self.model.model.layers[layer_idx].block_sparse_moe
			hook = moe.register_forward_hook(
				self._make_hook(f"mlp_{layer_idx}", scale=1.0, transpose=False)
			)
			self.hooks.append(hook)

	def collect(
		self,
		input_ids,
		position_ids=None,
		sequence_ids=None,
		attention_mask=None,
		cache_key=None,
	):
		"""
		input_ids: (B, T)
		Returns:
		x_stack_flat_SLH: (B*T, L+1, H) - embeddings + full layer outputs
		x_mlp_input_stack_flat_SLH: (B*T, L, H) - pre-LN residual stream (MLP inputs)
		x_mlp_stack_flat_SLH: (B*T, L, H) - MLP outputs
		x_clt_input_stack_flat_SLH: (B*T, L, H) - post-LN residual stream for CLT inference
		mask_S: (B*T,)
		cache_key: Unique identifier for the batch (e.g., a hash of input_ids).
		"""
		if isinstance(input_ids, dict):
			model_inputs = input_ids
			input_ids = model_inputs["input_ids"]
			position_ids = model_inputs["position_ids"]
			sequence_ids = model_inputs["sequence_ids"]
		else:
			if position_ids is None or sequence_ids is None:
				raise ValueError("position_ids and sequence_ids are required for ProGen3 collection")
			model_inputs = {
				"input_ids": input_ids,
				"position_ids": position_ids,
				"sequence_ids": sequence_ids,
			}

		if cache_key is not None and cache_key in self.cache:
			device = input_ids.device
			return tuple(
				t.to(device, non_blocking=True) if torch.is_tensor(t) else t
				for t in self.cache[cache_key]
			)

		self.activations.clear()
		with torch.no_grad():
			self.model(**model_inputs, return_dict=True)
		if not self.activations:
			raise RuntimeError("Collector failed: No activations captured.")
		if torch.cuda.is_available():
			torch.cuda.synchronize()

		B, T = input_ids.shape
		device = input_ids.device

		def get_flat_stack(keys, depth):
			if not keys:
				return None
			# Stack on CPU
			stack_cpu = torch.stack([self.activations[k] for k in keys], dim=1)
			# (B, L, T, H) -> (B, T, L, H) -> (S, L, H)
			# Move to GPU and cast back to float32 only for the specific batch being processed
			return stack_cpu.permute(0, 2, 1, 3).reshape(B * T, depth, -1).to(device=device, dtype=torch.float32)

		# 1. Main trajectory: embeddings + full layer outputs (integer keys)
		traj_keys = sorted([k for k in self.activations.keys() if isinstance(k, int)])
		x_stack_flat_SLH = get_flat_stack(traj_keys, len(traj_keys))

		# 2. MLP inputs (pre-LN residual stream, mlp_input_0, mlp_input_1, ...)
		mlp_in_keys = sorted(
			[k for k in self.activations.keys() if "mlp_input_" in str(k)],
			key=lambda x: int(str(x).split("_")[-1]),
		)
		x_mlp_input_stack_flat_SLH = get_flat_stack(mlp_in_keys, len(mlp_in_keys))

		# 3. MLP outputs (mlp_0, mlp_1, ...)
		mlp_keys = sorted(
			[k for k in self.activations.keys() if "mlp_" in str(k) and "input" not in str(k)],
			key=lambda x: int(str(x).split("_")[-1]),
		)
		x_mlp_stack_flat_SLH = get_flat_stack(mlp_keys, len(mlp_keys))

		# 4. CLT inputs (post-LN residual stream, clt_input_0, clt_input_1, ...)
		clt_keys = sorted(
			[k for k in self.activations.keys() if "clt_input_" in str(k)],
			key=lambda x: int(str(x).split("_")[-1]),
		)
		x_clt_input_stack_flat_SLH = get_flat_stack(clt_keys, len(clt_keys))

		# 5. Mask
		pad = self.vocabulary.get("<pad>")
		bos = self.vocabulary.get("<bos>")
		eos = self.vocabulary.get("<eos>")
		bos_glm = self.vocabulary.get("<bos_glm>")
		eos_span = self.vocabulary.get("<eos_span>")
		mask_tok = self.vocabulary.get("<mask>")
		one_tok = self.vocabulary.get("1")
		two_tok = self.vocabulary.get("2")

		mask_BT = torch.ones_like(input_ids, dtype=torch.bool)
		for tok in [pad, bos, eos, bos_glm, eos_span, mask_tok, one_tok, two_tok]:
			if tok is not None:
				mask_BT &= input_ids != tok
		mask_S = mask_BT.view(-1)

		results = (
			x_stack_flat_SLH,
			x_mlp_input_stack_flat_SLH,
			x_mlp_stack_flat_SLH,
			x_clt_input_stack_flat_SLH,
			mask_S,
		)

		if cache_key is not None:
			self.cache[cache_key] = tuple(
				t.to("cpu", non_blocking=True) if torch.is_tensor(t) else t for t in results
			)

		return results

	def remove_hooks(self):
		for h in self.hooks:
			h.remove()
		self.hooks = []


# Compatibility alias with requested naming style.
progen3activationcollector = ProGen3ActivationCollector


def main():
	"""Smoke-test ProGen3 activation capture for embeddings, MLP input/output, and CLT input."""
	import sys
	from pathlib import Path

	# Ensure imports resolve when executing this file directly.
	repo_root = Path(__file__).resolve().parents[1]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))
	progen3_src = repo_root / "external" / "progen3" / "src"
	if str(progen3_src) not in sys.path:
		sys.path.insert(0, str(progen3_src))

	from external.progen3.src.progen3.modeling import ProGen3ForCausalLM
	from external.progen3.src.progen3.batch_preparer import ProGen3BatchPreparer
	from training.clt_module import ProGen3ActivationCollector as LegacyProGen3ActivationCollector

	model_name = "Profluent-Bio/progen3-112m"
	sequence = (
		"MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVEL"
		"GGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
	)
	device = "cuda:0" if torch.cuda.is_available() else "cpu"

	print(f"[main] Loading model: {model_name}")
	print(f"[main] Using device: {device}")

	batch_preparer = ProGen3BatchPreparer()
	tokenizer = batch_preparer.tokenizer

	progen3_model = ProGen3ForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
	progen3_model = progen3_model.eval().to(device)
	for param in progen3_model.parameters():
		param.requires_grad = False

	collector = ProGen3ActivationCollector(progen3_model, tokenizer.get_vocab())
	collector.register_hooks()

	# Build model kwargs and run collection using the collector API.
	inputs = batch_preparer.get_batch_kwargs([sequence], device=device, reverse=False)
	cache_key = "progen3_activation_main_check"
	x_stack_flat_SLH, x_mlp_input_stack_flat_SLH, x_mlp_stack_flat_SLH, x_clt_input_stack_flat_SLH, mask_S = collector.collect(
		inputs,
		cache_key=cache_key,
	)

	num_layers = len(progen3_model.model.layers)
	B, T = inputs["input_ids"].shape

	print("[main] --- Captured Tensor Shapes ---")
	print("[main] x_stack_flat_SLH:", tuple(x_stack_flat_SLH.shape))
	print("[main] x_mlp_input_stack_flat_SLH:", tuple(x_mlp_input_stack_flat_SLH.shape))
	print("[main] x_mlp_stack_flat_SLH:", tuple(x_mlp_stack_flat_SLH.shape))
	print("[main] x_clt_input_stack_flat_SLH:", tuple(x_clt_input_stack_flat_SLH.shape))
	print("[main] mask_S:", tuple(mask_S.shape))

	# Structural checks: these ensure the hook wiring is coherent with the model depth.
	assert x_stack_flat_SLH.shape[0] == B * T, "x_stack batch-time dimension mismatch"
	assert x_stack_flat_SLH.shape[1] == num_layers + 1, "x_stack depth should be L+1 (embedding + layers)"

	assert x_mlp_input_stack_flat_SLH.shape[0] == B * T, "x_mlp_input batch-time dimension mismatch"
	assert x_mlp_input_stack_flat_SLH.shape[1] == num_layers, "x_mlp_input depth should be L"

	assert x_mlp_stack_flat_SLH.shape[0] == B * T, "x_mlp_output batch-time dimension mismatch"
	assert x_mlp_stack_flat_SLH.shape[1] == num_layers, "x_mlp_output depth should be L"

	assert x_clt_input_stack_flat_SLH.shape[0] == B * T, "x_clt_input batch-time dimension mismatch"
	assert x_clt_input_stack_flat_SLH.shape[1] == num_layers, "x_clt_input depth should be L"

	assert mask_S.shape[0] == B * T, "mask_S length should be B*T"

	# Optional sanity: output stacks should have non-zero magnitude in normal operation.
	print("[main] Mean |mlp_input|:", float(x_mlp_input_stack_flat_SLH.abs().mean().item()))
	print("[main] Mean |mlp_output|:", float(x_mlp_stack_flat_SLH.abs().mean().item()))
	print("[main] Mean |clt_input|:", float(x_clt_input_stack_flat_SLH.abs().mean().item()))

	# Compare against legacy collection path from training/clt_module.py.
	# Legacy "x" corresponds to block_sparse_moe input (post-LN), i.e. our clt_input stack.
	legacy_collector = LegacyProGen3ActivationCollector(progen3_model, tokenizer.get_vocab())
	legacy_collector.register_hooks()
	legacy_x_stack_flat_SLH, legacy_y_stack_flat_SLH, legacy_mask_S = legacy_collector.collect(
		batch_preparer,
		[sequence],
	)

	def nmse(pred, target, eps=1e-8):
		pred_f = pred.float()
		target_f = target.float()
		mse = torch.mean((pred_f - target_f) ** 2)
		den = torch.var(target_f, unbiased=False) + eps
		return mse / den

	nmse_mlp_input = nmse(x_clt_input_stack_flat_SLH, legacy_x_stack_flat_SLH)
	nmse_mlp_output = nmse(x_mlp_stack_flat_SLH, legacy_y_stack_flat_SLH)

	print("[main] --- Cross-check NMSE (new vs legacy) ---")
	print("[main] NMSE mlp_input (new clt_input vs legacy x):", float(nmse_mlp_input.item()))
	print("[main] NMSE mlp_output (new mlp_output vs legacy y):", float(nmse_mlp_output.item()))

	assert legacy_mask_S.shape == mask_S.shape, "legacy mask shape mismatch"
	# "Near 0" with small numerical tolerance for fused kernels / dtype conversions.
	assert float(nmse_mlp_input.item()) < 1e-3, "MLP input NMSE is not near 0"
	assert float(nmse_mlp_output.item()) < 1e-3, "MLP output NMSE is not near 0"

	legacy_collector.remove_hooks()

	collector.remove_hooks()
	print("[main] ProGen3ActivationCollector check passed.")


if __name__ == "__main__":
	main()
