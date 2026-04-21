"""
Plot total cross-section σ vs number of final-state particles
for φ³, Yukawa, and QED theories using RAMBO Monte Carlo integration.

Particle choices (all produce bosons in the final state):
  φ³    : φφ  → n × φ      (scalars  → scalars)
  Yukawa: ψψ̄ → n × φ      (fermions → scalars)
  QED   : ψψ̄ → n × γ      (fermions → photons)

Final-state counts: n = 2, 3, 4, 5, 6.
Errors estimated by running n_batches independent MC runs.

NOTE: currents_qed_v2.spin_averaged_matrix_element expects masses_psi
as a dict  {"flavour": mass}, but mc_integration passes a bare float.
This script bypasses mc_cross_section for QED to handle that correctly.
"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import scipy.special as sp
import mc_integration as mc

from currents_phi import calculate_matrix_element as phi_matrix_element
from currents_yukawa import FieldType as YFT, spin_averaged_matrix_element as yukawa_matrix_element
from currents_qed_v2 import FieldType as QFT, spin_averaged_matrix_element as qed_matrix_element

# ─── Parameters ───────────────────────────────────────────────────────
com_energy  = 100.0
vev         = 246.21965
f_s_constant= 137.035999084

m_higgs     = 125.2

# Explicitly used couplings and masses for the three theories:
m_phi       = 5     #arbitrary light scalar mass for φ³ and Yukawa theories
m_psi_benchmark = 5 # benchmark fermion mass for Yukawa coupling and QED (e.g. electron mass)
masses_psi  = {"electron": m_psi_benchmark}
#masses_psi  = {"electron": 0.00051099895, "muon": 0.1057583755, "tau": 1.77693}

e           = np.sqrt(4 * np.pi / f_s_constant)
g           = e
lambda_0    = e # match QED strength


n_final_list        = [2, 3, 4, 5, 6]
qed_n_final_list    = [2, 3, 4, 5]  # n=6 too slow for QED with current settings

# ─── Generic MC cross-section (one batch) ────────────────────────────
# Reimplemented here so we can pass masses_psi as a dict for QED
# while keeping a float for Yukawa, without touching mc_integration.py.

def mc_sigma_one_batch(com_energy, no_events, no_particles,
                       theory, theory_kwargs):
    """
    Single-batch total cross-section via RAMBO.
    theory_kwargs holds everything the matrix-element function needs.
    """
    masses = theory_kwargs.get("masses")
    masses_in, masses_out, no_outgoing = mc.separate_in_and_out(no_particles, masses)

    total = 0.0
    for _ in range(no_events):
        p_in  = mc.incoming_momenta(com_energy, masses_in)
        p_out, wt = mc.generate_phase_space(com_energy, no_outgoing, masses_out)
        p_event = np.vstack((*p_in, p_out))

        if theory == "phi":
            me_sq = phi_matrix_element(
                p_event,
                theory_kwargs["m_phi"],
                theory_kwargs["lambda_0"])

        elif theory == "yukawa":
            pts = _build_particle_list(p_event, theory_kwargs)
            me_sq = yukawa_matrix_element(
                pts,
                theory_kwargs["m_phi"],
                theory_kwargs["masses_psi"],   # dict
                theory_kwargs["g"])

        elif theory == "qed":
            pts = _build_particle_list(p_event, theory_kwargs)
            me_sq = qed_matrix_element(
                pts,
                theory_kwargs["masses_psi"],  # dict
                theory_kwargs["e"])

        total += me_sq * wt

    weight_0 = ((np.pi / 2) ** (no_outgoing - 1)
                * com_energy ** (2 * no_outgoing - 4)
                / (sp.gamma(no_outgoing) * sp.gamma(no_outgoing - 1)))
    flux = mc.flux_factor(com_energy, masses_in)
    return total * weight_0 / (no_events * flux)


def _build_particle_list(p_event, kw):
    """Build the list-of-dicts that Yukawa / QED matrix elements expect."""
    field_types = kw["field_types"]
    flavours    = kw["flavours"]
    n = len(field_types)
    pts = [{"type": field_types[0], "p": p_event[0],
            "incoming": True,  "flavour": flavours[0]},
           {"type": field_types[1], "p": p_event[1],
            "incoming": True,  "flavour": flavours[1]}]
    for j in range(2, n):
        pts.append({"type": field_types[j], "p": p_event[j],
                     "incoming": False, "flavour": flavours[j]})
    return pts


# ── File I/O ────────────────────────────────────────────────────────
RESULTS_FILE = "sigma_results.txt"
 
def init_results_file(filepath=RESULTS_FILE, com_energy=None):
    """Write header. Call once at the start of a run."""
    with open(filepath, "w") as f:
        f.write(f"# sigma vs n_final results\n")
        f.write(f"# com_energy = {com_energy}\n")
        f.write(f"# generated  = {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#\n")
        f.write(f"# {'theory':<10s} {'n_final':>7s} {'sigma':>14s} "
                f"{'stderr':>14s} {'rel_err%':>9s} {'n_batches':>9s} "
                f"{'n_events':>9s} {'time_min':>9s}\n")
    print(f"Results file initialised: {filepath}\n")
 
 
def append_result(theory, n_final, mean, stderr, n_batches, n_events,
                  elapsed_s, filepath=RESULTS_FILE):
    """Append one data point. Called after each (theory, n) completes."""
    rel_err = abs(stderr / mean) * 100 if mean != 0 else float('inf')
    with open(filepath, "a") as f:
        f.write(f"  {theory:<10s} {n_final:>7d} {mean:>14.6e} "
                f"{stderr:>14.6e} {rel_err:>9.1f} {n_batches:>9d} "
                f"{n_events:>9d} {elapsed_s/60:>9.1f}\n")
 
 
def load_results(filepath=RESULTS_FILE):
    """
    Read saved results back into a dict for plotting.
 
    Returns:
    --------
    dict  :  {theory: {"n": [...], "sigma": [...], "stderr": [...]}}
    """
    data = {}
    with open(filepath, "r") as f:
        for line in f:
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
 
    # Convert to arrays
    for theory in data:
        for key in data[theory]:
            data[theory][key] = np.array(data[theory][key])
    return data


# ── Adaptive MC integration ────────────────────────────────────────
def sigma_adaptive(n_final, theory, theory_kwargs,
                   com_energy, events_per_batch=5000,
                   target_rel_error=0.10, min_batches=5,
                   max_batches=500, max_time_s=None,
                   save=True, results_file=RESULTS_FILE):
    """
    Run MC batches until relative uncertainty < target OR time runs out.
    Automatically appends the result to results_file when done.
 
    Returns:
    --------
    mean, stderr, n_batches_used, total_events
    """
    no_particles = n_final + 2
    batch_results = []
    t_start = time.perf_counter()
 
    for b in range(1, max_batches + 1):
        sig = mc_sigma_one_batch(com_energy, events_per_batch,
                                 no_particles, theory, theory_kwargs)
        batch_results.append(sig)
        elapsed = time.perf_counter() - t_start
 
        if b >= min_batches:
            arr    = np.array(batch_results)
            mean   = arr.mean()
            stderr = arr.std(ddof=1) / np.sqrt(b)
            rel_err = abs(stderr / mean) if mean != 0 else float('inf')
 
            print(f"  {theory:8s} n={n_final}  batch {b:>4d}  "
                  f"σ = {mean:.4e} ± {stderr:.4e}  "
                  f"({rel_err*100:.1f}%)  [{elapsed/60:.1f} min]")
 
            if rel_err < target_rel_error:
                print(f"  ✓ Converged: {rel_err*100:.1f}% < "
                      f"{target_rel_error*100:.0f}% target "
                      f"({b * events_per_batch} events, "
                      f"{elapsed/60:.1f} min)")
                if save:
                    append_result(theory, n_final, mean, stderr,
                                 b, b * events_per_batch, elapsed,
                                 results_file)
                return mean, stderr, b, b * events_per_batch
 
            if max_time_s is not None and elapsed > max_time_s:
                print(f"  ⏱ Time cap reached ({elapsed/60:.1f} min). "
                      f"Best: {rel_err*100:.1f}%")
                if save:
                    append_result(theory, n_final, mean, stderr,
                                 b, b * events_per_batch, elapsed,
                                 results_file)
                return mean, stderr, b, b * events_per_batch
        else:
            print(f"  {theory:8s} n={n_final}  batch {b:>4d}  "
                  f"σ = {sig:.4e}  (warming up)  "
                  f"[{elapsed/60:.1f} min]")
 
            if max_time_s is not None and elapsed > max_time_s:
                arr    = np.array(batch_results)
                mean   = arr.mean()
                stderr = arr.std(ddof=1) / np.sqrt(b) if b > 1 else float('inf')
                rel_err = abs(stderr / mean) if (b > 1 and mean != 0) else float('inf')
                print(f"  ⏱ Time cap during warmup ({elapsed/60:.1f} min). "
                      f"{b} batches, ~{rel_err*100:.0f}% error")
                if save:
                    append_result(theory, n_final, mean, stderr,
                                 b, b * events_per_batch, elapsed,
                                 results_file)
                return mean, stderr, b, b * events_per_batch
 
    arr    = np.array(batch_results)
    mean   = arr.mean()
    stderr = arr.std(ddof=1) / np.sqrt(len(arr))
    rel_err = abs(stderr / mean) if mean != 0 else float('inf')
    elapsed = time.perf_counter() - t_start
    print(f"  ⚠ Max batches. Relative error: {rel_err*100:.1f}%")
    if save:
        append_result(theory, n_final, mean, stderr,
                     max_batches, max_batches * events_per_batch,
                     elapsed, results_file)
    return mean, stderr, max_batches, max_batches * events_per_batch
 
 
# ── Per-(theory, n_final) settings ──────────────────────────────────
 
settings = {
    "phi": {
        2: {"events_per_batch":  5000, "target": 0.10, "max_time_s":   120, "min_batches": 5},
        3: {"events_per_batch":  5000, "target": 0.10, "max_time_s":   120, "min_batches": 5},
        4: {"events_per_batch":  5000, "target": 0.10, "max_time_s":   120, "min_batches": 5},
        5: {"events_per_batch":  5000, "target": 0.10, "max_time_s":   300, "min_batches": 5},
        6: {"events_per_batch":  5000, "target": 0.10, "max_time_s":   300, "min_batches": 5},
    },
    "yukawa": {
        2: {"events_per_batch":  5000, "target": 0.10, "max_time_s":   120, "min_batches": 5},
        3: {"events_per_batch":  5000, "target": 0.10, "max_time_s":   120, "min_batches": 5},
        4: {"events_per_batch":  5000, "target": 0.10, "max_time_s":   300, "min_batches": 5},
        5: {"events_per_batch":  5000, "target": 0.10, "max_time_s":   600, "min_batches": 5},
        6: {"events_per_batch":  5000, "target": 0.10, "max_time_s":   300, "min_batches": 5},
    },
    "qed": {
        2: {"events_per_batch": 20000, "target": 0.10, "max_time_s":   600, "min_batches": 5},
        3: {"events_per_batch": 20000, "target": 0.15, "max_time_s":  5400, "min_batches": 5},
        4: {"events_per_batch":  5000, "target": 0.25, "max_time_s": 10800, "min_batches": 3},
        5: {"events_per_batch":  1000, "target": 0.40, "max_time_s": 18000, "min_batches": 3},
    },
}

init_results_file(com_energy=com_energy)
# ═══════════════════════════════════════════════════════════════════════
#  φ³ :  φφ → n × φ   (all scalars, mass m_phi)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60, "\nφ³ theory\n" + "=" * 60)
phi3_e_means, phi3_e_errs = [], []
for n in n_final_list:
    kw = {
        "masses":   np.array([m_phi] * (n + 2)),
        "m_phi":    m_phi,
        "lambda_0": lambda_0}
    s = settings["phi"][n]
    mean, err, _, _ = sigma_adaptive(
        n, "phi", kw, com_energy,
        events_per_batch = s["events_per_batch"],
        target_rel_error = s["target"],
        min_batches      = s["min_batches"],
        max_time_s       = s["max_time_s"],
    )
    phi3_e_means.append(mean)
    phi3_e_errs.append(err)
    print(f"  → σ = {mean:.6e} ± {err:.6e}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Yukawa :  ψψ̄ → n × φ   (fermions → scalars)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60, "\nYukawa theory\n" + "=" * 60)
yuk_means, yuk_errs = [], []
for n in n_final_list:
    kw = {"masses":      np.array([masses_psi["electron"], masses_psi["electron"]] + [m_phi] * n),
          "field_types": [YFT.PSI, YFT.PSIBAR] + [YFT.PHI] * n,
          "flavours":    ["electron", "electron"] + [None] * n,
          "m_phi":       m_phi,
          "masses_psi":  masses_psi,
          "g":           g}
    s = settings["yukawa"][n]
    mean, err, _, _ = sigma_adaptive(
        n, "yukawa", kw, com_energy,
        events_per_batch = s["events_per_batch"],
        target_rel_error = s["target"],
        min_batches      = s["min_batches"],
        max_time_s       = s["max_time_s"],
         )
    yuk_means.append(mean)
    yuk_errs.append(err)
    print(f"  → σ = {mean:.6e} ± {err:.6e}\n")


# ═══════════════════════════════════════════════════════════════════════
#  QED :  ψψ̄ → n × γ   (fermions → photons, massless final state)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60, "\nQED theory\n" + "=" * 60)
qed_means, qed_errs = [], []
for n in qed_n_final_list:
    # Photons are massless; incoming fermions have mass in massees_psi by flavour
    kw = {"masses":     np.array([masses_psi["electron"], masses_psi["electron"]] + [0.0] * n),
          "field_types":[QFT.PSI, QFT.PSIBAR] + [QFT.A] * n,
          "flavours":   ["electron", "electron"] + [None] * n,
          "masses_psi": masses_psi,   # dict keyed by flavour
          "e":          e}
    s = settings["qed"][n]
    mean, err, _, _ = sigma_adaptive(
        n, "qed", kw, com_energy,
        events_per_batch = s["events_per_batch"],
        target_rel_error = s["target"],
        min_batches      = s["min_batches"],
        max_time_s       = s["max_time_s"],
         )
    qed_means.append(mean)
    qed_errs.append(err)
    print(f"  → σ = {mean:.6e} ± {err:.6e}\n")