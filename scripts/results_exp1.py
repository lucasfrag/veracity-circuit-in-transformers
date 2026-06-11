import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

aie_resid = np.load("results/aie_resid.npy")
aie_attn  = np.load("results/aie_attn.npy")
aie_mlp   = np.load("results/aie_mlp.npy")

n_layers = len(aie_resid)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Causal Tracing — Llama 3.1 8B (N=176 fatos)", fontsize=13)

components = [
    (aie_resid, "Residual Stream (resid_pre)", "Blues"),
    (aie_attn,  "Attention Output (attn_out)", "Oranges"),
    (aie_mlp,   "MLP Output (mlp_out)",        "Greens"),
]

for ax, (aie, title, cmap) in zip(axes, components):
    im = ax.imshow(
        aie[np.newaxis, :],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=aie.clip(0).max(),
    )
    ax.set_yticks([])
    ax.set_ylabel(title, fontsize=9, rotation=0, labelpad=160, va="center")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

axes[-1].set_xlabel("Layer", fontsize=10)
plt.tight_layout()
plt.savefig("results/causal_trace_heatmap_full.png", dpi=150, bbox_inches="tight")
print("Salvo em results/causal_trace_heatmap_full.png")

print(f"\nAIE por layer — mlp_out:")
for i, v in enumerate(aie_mlp):
    bar = "█" * int(v * 400)
    print(f"  layer {i:2d}: {v:.4f} {bar}")