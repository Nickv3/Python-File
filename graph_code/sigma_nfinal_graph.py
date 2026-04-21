"""
Plot sigma vs n_final from saved results file.
 
Run this standalone after the main computation finishes (or even
while it's still running — it plots whatever data exists so far).
 
Usage:
    python plot_sigma_results.py
    python plot_sigma_results.py my_results.txt     # custom file
"""
 
import sys
import numpy as np
import matplotlib.pyplot as plt
 
 
def load_results(filepath="data/sigma_results.txt"):
    """
    Read saved results into a dict for plotting.
    Returns:  {theory: {"n": array, "sigma": array, "stderr": array}}
    """
    data = {}
    com_energy = None
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("# com_energy"):
                com_energy = float(line.split("=")[1].strip())
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            theory  = parts[0]
            n_final = int(parts[1])
            sigma   = float(parts[2])
            stderr  = float(parts[3])
 
            if theory not in data:
                data[theory] = {"n": [], "sigma": [], "stderr": []}
            data[theory]["n"].append(n_final)
            data[theory]["sigma"].append(sigma)
            data[theory]["stderr"].append(stderr)
 
    for theory in data:
        for key in data[theory]:
            data[theory][key] = np.array(data[theory][key])
    return data, com_energy
 
 
def plot_results(filepath="data/sigma_results.txt", output="sigma_vs_nfinal.png"):
    data, com_energy = load_results(filepath)
 
    if not data:
        print(f"No data found in {filepath}")
        return
 
    print(f"Loaded data for: {list(data.keys())}")
    for theory in data:
        print(f"  {theory}: n = {data[theory]['n'].tolist()}")
 
    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6.5))
 
    plot_config = {
        "yukawa": {"fmt": "s-", "color": "#ff7f0e",
                "label": r"Yukawa: $\psi\bar\psi \to n\,\phi$"},
        "qed":    {"fmt": "^-", "color": "#2ca02c",
                "label": r"QED: $\psi\bar\psi \to n\,\gamma$"},}
    phi_styles = [
    {"color": "#1f77b4", "label": r"$\phi^3(g = e)$: $\phi\phi\to n\,\phi$", "fmt": "o-"},
    {"color": "#b41fa3", "label": r"$\phi^3(g = y_f)$: $\phi\phi\to n\,\phi$", "fmt": "--"},
    {"color": "#17becf", "label": r"$\phi^3(alt)$: $\phi\phi\to n\,\phi$", "fmt": "o:"},]
    phi_keys = [k for k in data if k.startswith("phi")]
 
    offsets = {"phi_e": -0.06, "phi_y": 0.0, "yukawa": 0.0, "qed": 0.06}
 
    # First plot φ³ variants
    for i, theory in enumerate(sorted(phi_keys)):
        d = data[theory]
        style = phi_styles[i % len(phi_styles)]
        label=f"{style['label']}"

        ax.errorbar(d["n"] - 0.06 + 0.06*i, d["sigma"],
                    yerr=d["stderr"],
                    fmt=style["fmt"], capsize=5, markersize=8, linewidth=2,
                    color=style["color"],
                    label=label)

    # Then plot Yukawa + QED
    for theory in ["yukawa", "qed"]:
        if theory not in data:
            continue
        d   = data[theory]
        cfg = plot_config[theory]

        offset = 0.06 if theory == "qed" else 0.0

        ax.errorbar(d["n"] + offset, d["sigma"],
                    yerr=d["stderr"],
                    fmt=cfg["fmt"], capsize=5, markersize=8, linewidth=2,
                    color=cfg["color"], label=cfg["label"])
 
    ax.set_xlabel("Final-state particles,  $n$", fontsize=14)
    ax.set_ylabel(r"Cross section, $\sigma$", fontsize=14)
 
    title = r"$\sigma$ vs $n_{\mathrm{final}}$"
    if com_energy:
        title += f"  ($\\sqrt{{s}} = {com_energy:.0f}$ GeV)"
    #ax.set_title(title, fontsize=14)
 
    ax.set_yscale("log")
    all_n = np.concatenate([data[t]["n"] for t in data])
    ax.set_xticks(sorted(set(all_n.astype(int))))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(fontsize=12)
    #ax.grid(False, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"\nPlot saved to {output}")
    plt.show()
 
 
if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/sigma_results.txt"
    plot_results(filepath)