import torch
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(
    "meta-llama/Meta-Llama-3.1-8B",
    device="cuda",
    dtype=torch.bfloat16,
)

# Inspecionar as chaves do cache
logits, cache = bridge.run_with_cache("The mother tongue of Danielle Darrieux is")

keys = list(cache.keys())
print(f"Total de hooks: {len(keys)}")
print("\nPrimeiros 30 hooks:")
for k in keys[:30]:
    print(f"  {k}")

print("\n...")
print("\nÚltimos 10 hooks:")
for k in keys[-10:]:
    print(f"  {k}")