import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformer_lens.model_bridge import TransformerBridge

# ── Configuração ──────────────────────────────────────────────────────────────
MODEL_NAME         = "meta-llama/Meta-Llama-3.1-8B"
NOISE_SIGMA_FACTOR = 3.0
N_ENTRIES          = 183
BATCH_SIZE         = 10   # salva checkpoint a cada 10 fatos
CHECKPOINT_DIR     = Path("results/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Carrega modelo ─────────────────────────────────────────────────────────────
print("Carregando modelo...")
bridge = TransformerBridge.boot_transformers(
    MODEL_NAME, device="cuda", dtype=torch.bfloat16,
)
bridge.eval()
n_layers = bridge.cfg.n_layers
print(f"Modelo carregado — {n_layers} layers\n")

# ── Carrega CounterFact ────────────────────────────────────────────────────────
with open("data/counterfact/counterfact.json") as f:
    counterfact = json.load(f)
with open("data/counterfact/known_ids.json") as f:
    known_ids = set(json.load(f))

known = [e for e in counterfact if e["case_id"] in known_ids][:N_ENTRIES]
print(f"Total de fatos para processar: {len(known)}\n")

# ── Verifica checkpoint existente ──────────────────────────────────────────────
checkpoint_file = CHECKPOINT_DIR / "progress.json"

if checkpoint_file.exists():
    with open(checkpoint_file) as f:
        progress = json.load(f)
    start_idx   = progress["next_idx"]
    n_valid     = progress["n_valid"]
    aie_resid   = np.array(progress["aie_resid"])
    aie_attn    = np.array(progress["aie_attn"])
    aie_mlp     = np.array(progress["aie_mlp"])
    print(f"Checkpoint encontrado — retomando do fato {start_idx} ({n_valid} já processados)\n")
else:
    start_idx = 0
    n_valid   = 0
    aie_resid = np.zeros(n_layers)
    aie_attn  = np.zeros(n_layers)
    aie_mlp   = np.zeros(n_layers)
    print("Nenhum checkpoint encontrado — iniciando do zero\n")

# ── Estima sigma do ruído ──────────────────────────────────────────────────────
embed_capture = {}
def capture_embed(value, hook):
    embed_capture["std"] = value.float().std().item()
    return value

with torch.no_grad():
    bridge.run_with_hooks("test", fwd_hooks=[("hook_embed", capture_embed)])

noise_sigma = NOISE_SIGMA_FACTOR * embed_capture["std"]
print(f"Sigma do ruído: {noise_sigma:.6f}\n")

# ── Funções auxiliares ─────────────────────────────────────────────────────────
def get_subject_positions(bridge, prompt, subject):
    tokens_full    = bridge.to_tokens(prompt, prepend_bos=True)[0]
    tokens_subject = bridge.to_tokens(" " + subject, prepend_bos=False)[0]
    n = len(tokens_subject)
    for start in range(len(tokens_full) - n + 1):
        if torch.equal(tokens_full[start:start + n], tokens_subject):
            return list(range(start, start + n))
    tokens_subject2 = bridge.to_tokens(subject, prepend_bos=False)[0]
    n2 = len(tokens_subject2)
    for start in range(len(tokens_full) - n2 + 1):
        if torch.equal(tokens_full[start:start + n2], tokens_subject2):
            return list(range(start, start + n2))
    return []

def get_target_token(bridge, target_str):
    return bridge.to_tokens(" " + target_str, prepend_bos=False)[0, 0]

def prob_of_token(logits, token_id):
    return torch.softmax(logits[0, -1].float(), dim=-1)[token_id].item()

def make_corrupt(pos, sigma):
    def hook(value, hook):
        value[:, pos, :] += torch.randn_like(value[:, pos, :]) * sigma
        return value
    return hook

def make_restore(pos, clean_val):
    def hook(value, hook):
        value[:, pos, :] = clean_val[:, pos, :]
        return value
    return hook

def make_capture(storage, key):
    def hook(value, hook):
        storage[key] = value.detach().clone()
        return value
    return hook

def save_checkpoint(next_idx, n_valid, aie_resid, aie_attn, aie_mlp):
    progress = {
        "next_idx":  next_idx,
        "n_valid":   n_valid,
        "aie_resid": aie_resid.tolist(),
        "aie_attn":  aie_attn.tolist(),
        "aie_mlp":   aie_mlp.tolist(),
    }
    with open(checkpoint_file, "w") as f:
        json.dump(progress, f)

# ── Loop principal ─────────────────────────────────────────────────────────────
remaining = known[start_idx:]

for batch_start in range(0, len(remaining), BATCH_SIZE):
    batch = remaining[batch_start:batch_start + BATCH_SIZE]

    for entry in tqdm(
        batch,
        desc=f"Fatos {start_idx + batch_start + 1}–"
             f"{min(start_idx + batch_start + BATCH_SIZE, len(known))} / {len(known)}"
    ):
        rw       = entry["requested_rewrite"]
        prompt   = rw["prompt"].format(rw["subject"])
        subject  = rw["subject"]
        tok_true = get_target_token(bridge, rw["target_true"]["str"])

        subj_pos = get_subject_positions(bridge, prompt, subject)
        if not subj_pos:
            tqdm.write(f"  Subject não encontrado: {subject}")
            continue

        # Passo 1: clean run
        with torch.no_grad():
            logits_clean = bridge.run_with_hooks(prompt, fwd_hooks=[])
        p_clean = prob_of_token(logits_clean, tok_true)

        # Passo 2: corrupted run
        with torch.no_grad():
            logits_corrupt = bridge.run_with_hooks(
                prompt,
                fwd_hooks=[("hook_embed", make_corrupt(subj_pos, noise_sigma))]
            )
        p_corrupt = prob_of_token(logits_corrupt, tok_true)

        if abs(p_clean - p_corrupt) < 1e-4:
            tqdm.write(f"  Ruído sem efeito: {subject}")
            continue

        # Passo 3: restore runs por layer
        for layer in range(n_layers):
            clean = {}
            with torch.no_grad():
                bridge.run_with_hooks(
                    prompt,
                    fwd_hooks=[
                        (f"blocks.{layer}.hook_resid_pre",
                         make_capture(clean, "resid_pre")),
                        (f"blocks.{layer}.hook_attn_out",
                         make_capture(clean, "attn_out")),
                        (f"blocks.{layer}.hook_mlp_out",
                         make_capture(clean, "mlp_out")),
                    ]
                )

            for key, aie_arr, hook_name in [
                ("resid_pre", aie_resid, f"blocks.{layer}.hook_resid_pre"),
                ("attn_out",  aie_attn,  f"blocks.{layer}.hook_attn_out"),
                ("mlp_out",   aie_mlp,   f"blocks.{layer}.hook_mlp_out"),
            ]:
                with torch.no_grad():
                    logits_r = bridge.run_with_hooks(
                        prompt,
                        fwd_hooks=[
                            ("hook_embed",
                             make_corrupt(subj_pos, noise_sigma)),
                            (hook_name,
                             make_restore(subj_pos, clean[key])),
                        ]
                    )
                aie_arr[layer] += prob_of_token(logits_r, tok_true) - p_corrupt

            del clean
            torch.cuda.empty_cache()

        n_valid += 1

    # Salva checkpoint após cada batch
    current_idx = start_idx + batch_start + len(batch)
    save_checkpoint(current_idx, n_valid, aie_resid, aie_attn, aie_mlp)
    print(f"\nCheckpoint salvo — {n_valid} fatos processados até agora\n")

# ── Resultado final ────────────────────────────────────────────────────────────
if n_valid > 0:
    aie_resid_avg = aie_resid / n_valid
    aie_attn_avg  = aie_attn  / n_valid
    aie_mlp_avg   = aie_mlp   / n_valid
else:
    aie_resid_avg = aie_resid
    aie_attn_avg  = aie_attn
    aie_mlp_avg   = aie_mlp

np.save("results/aie_resid.npy", aie_resid_avg)
np.save("results/aie_attn.npy",  aie_attn_avg)
np.save("results/aie_mlp.npy",   aie_mlp_avg)

print(f"Fatos processados: {n_valid}")
print(f"\nAIE médio por componente:")
print(f"  resid_pre : max={aie_resid_avg.max():.4f} @ layer {aie_resid_avg.argmax()}")
print(f"  attn_out  : max={aie_attn_avg.max():.4f}  @ layer {aie_attn_avg.argmax()}")
print(f"  mlp_out   : max={aie_mlp_avg.max():.4f}   @ layer {aie_mlp_avg.argmax()}")

print("\nExperimento concluído.")