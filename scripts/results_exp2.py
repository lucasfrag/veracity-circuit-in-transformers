import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Experimento 1 — claims verdadeiros
aie_resid_true = np.load("results/aie_resid_true.npy")
aie_attn_true  = np.load("results/aie_attn_true.npy")
aie_mlp_true   = np.load("results/aie_mlp_true.npy")

# Experimento 2 — claims contrafactuais
aie_resid_cf = np.load("results/aie_resid_cf.npy")
aie_attn_cf  = np.load("results/aie_attn_cf.npy")
aie_mlp_cf   = np.load("results/aie_mlp_cf.npy")

n_layers = len(aie_resid_true)
layers   = np.arange(n_layers)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(
    "Causal Tracing — Claims Verdadeiros vs. Contrafactuais\n"
    "Llama 3.1 8B",
    fontsize=13
)

components = [
    (aie_resid_true, aie_resid_cf, "Residual Stream (resid_pre)", "#1f77b4", "#aec7e8"),
    (aie_attn_true,  aie_attn_cf,  "Attention Output (attn_out)", "#d62728", "#f4a582"),
    (aie_mlp_true,   aie_mlp_cf,   "MLP Output (mlp_out)",        "#2ca02c", "#98df8a"),
]

for ax, (true_vals, cf_vals, title, color_true, color_cf) in zip(axes, components):
    ax.bar(layers - 0.2, true_vals, width=0.4,
           label="Verdadeiro", color=color_true, alpha=0.9)
    ax.bar(layers + 0.2, cf_vals,   width=0.4,
           label="Contrafactual", color=color_cf, alpha=0.9)
    ax.set_ylabel("AIE médio", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

axes[-1].set_xlabel("Layer", fontsize=10)
plt.tight_layout()
plt.savefig("results/exp1_vs_exp2_comparison.png", dpi=150, bbox_inches="tight")
print("Salvo em results/exp1_vs_exp2_comparison.png")

# Imprime diferenças chave
print("\nDiferença (Verdadeiro - Contrafactual) por componente:")
print(f"  resid_pre : max diff = {(aie_resid_true - aie_resid_cf).max():.4f} "
      f"@ layer {(aie_resid_true - aie_resid_cf).argmax()}")
print(f"  attn_out  : max diff = {(aie_attn_true - aie_attn_cf).max():.4f} "
      f"@ layer {(aie_attn_true - aie_attn_cf).argmax()}")
print(f"  mlp_out   : max diff = {(aie_mlp_true - aie_mlp_cf).max():.4f} "
      f"@ layer {(aie_mlp_true - aie_mlp_cf).argmax()}")