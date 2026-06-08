import json
import torch
from tqdm import tqdm
from pathlib import Path
from transformer_lens.model_bridge import TransformerBridge

# Carrega o modelo
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
print(f"Carregando {MODEL_NAME}...")
bridge = TransformerBridge.boot_transformers(
    MODEL_NAME,
    device="cuda",
    dtype=torch.bfloat16,
)
print("Modelo carregado!\n")

# Carrega o CounterFact
with open("data/counterfact/counterfact.json") as f:
    counterfact = json.load(f)
print(f"Total de entradas no CounterFact: {len(counterfact)}")

# Função de verificação
def check_fact(bridge, entry):
    rw = entry["requested_rewrite"]
    prompt = rw["prompt"].format(rw["subject"])
    target_true = " " + rw["target_true"]["str"]
    target_new  = " " + rw["target_new"]["str"]

    tok_true = bridge.to_tokens(target_true, prepend_bos=False)[0, 0]
    tok_new  = bridge.to_tokens(target_new,  prepend_bos=False)[0, 0]

    with torch.no_grad():
        logits, _ = bridge.run_with_cache(prompt)

    last_logits = logits[0, -1]
    score_true = last_logits[tok_true].item()
    score_new  = last_logits[tok_new].item()

    return score_true > score_new, score_true, score_new

# Avaliar nas primeiras 200 entradas
SAMPLE = counterfact[:200]
results = []

for entry in tqdm(SAMPLE, desc="Verificando fatos"):
    try:
        correct, st, sn = check_fact(bridge, entry)
        results.append({"correct": correct, "score_true": st, "score_new": sn})
    except Exception as e:
        results.append({"correct": False, "score_true": 0.0, "score_new": 0.0})

n_correct = sum(r["correct"] for r in results)
print(f"\nModelo conhece o fato: {n_correct}/{len(results)} ({100*n_correct/len(results):.1f}%)")

# Salva os IDs dos fatos conhecidos
known_ids = [
    counterfact[i]["case_id"]
    for i, r in enumerate(results)
    if r["correct"]
]
out_path = Path("data/counterfact/known_ids.json")
with open(out_path, "w") as f:
    json.dump(known_ids, f)

print(f"IDs salvos em {out_path}")