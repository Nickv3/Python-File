import mc_integration as mc
import observables as obs

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

from currents_phi import calculate_matrix_element
from currents_yukawa import FieldType as YFT, spin_averaged_matrix_element as yukawa_matrix_element
from currents_qed_v2 import FieldType as QFT, spin_averaged_matrix_element as qed_matrix_element



def differential_cross_section(com_energy:float, no_events:int, no_particles:int, constants, theory, field_types = None, masses = None, flavours = None, obs_fun = None):
    """
    Produces a differential cross-section graph by observable 'obs_fun' in scattering of no_particles massless particles using Monte Carlo integration and rambo phase space generator.

    Parameters:
    com_energy (float): COM energy.
    no_events (int): Number of Monte Carlo events to generate.
    no_particles (int): Number of particles involved in scattering.

    constants (dict): Dictionary of theory constants, m_phi, masses_psi, lambda_0, g, e.
    theory (str): "phi", "yukawa" or "qed" to specify which theory to use for matrix element calculation.
    field_types (list): List of particle types (e.g. "scalar", "fermion", "photon") for each particle, used for matrix element calculation. Not used in "phi".
    masses (np.nd.array): 1D array of masses of n particles. Either m_phi or masses_psi[flavour] for "phi" and "yukawa", and masses_psi[flavour] or 0 for "qed". If all massless set to None.
    flavours (list): List of particle flavours for each particle, used for matrix element calculation. Not used in "phi" or "yukawa".
    obs_fun (function): Function of observable to measure/plot.

    Returns:
    cross_section (float): Estimated cross-section.
    """
    total_weighted_me_sq = 0 # Running total of w_i * |M(p_i)|^2
    obs_values = []
    cross_section_values = []
    
    # Separate incoming and outgoing particle masses and numbers
    masses_in, masses_out, no_outgoing = mc.separate_in_and_out(no_particles, masses)

    # Calculate weighted sum of matrix element squared * observable value
    for i in range(no_events):
        if i % 100 == 0:
            print(f"Processing event {i+1}/{no_events}...")
        # incoming momenta (2 particles)
        p_in = mc.incoming_momenta(com_energy, masses_in)

        # outgoing momenta (n particles), i.e. the phase space points
        p_out, weight_m = mc.generate_phase_space(com_energy, no_outgoing, masses_out)

        # full event momenta: [p1, p2, p3, ..., p_{n+2}]
        p_event = np.vstack((*p_in, p_out))
        #if i < 10:
        #    print(p_event)

        if theory == "phi":
            me_sq = calculate_matrix_element(p_event, constants['m_phi'], constants['lambda_0'])

        elif theory == "yukawa":
            p_event_type = [{"type": field_types[0], "p": p_event[0], "incoming": True, "flavour": flavours[0]},
                            {"type": field_types[1], "p": p_event[1], "incoming": True, "flavour": flavours[1]},]
            for j in range(2, no_particles):
                p_event_type.append({"type": field_types[j], "p": p_event[j], "incoming": False, "flavour": flavours[j]})
            me_sq = yukawa_matrix_element(p_event_type, constants['m_phi'], constants['masses_psi'], constants['g'])

        elif theory == "qed":
            p_event_type = [{"type": field_types[0], "p": p_event[0], "incoming": True, "flavour": flavours[0]},
                            {"type": field_types[1], "p": p_event[1], "incoming": True, "flavour": flavours[1]},]
            for j in range(2, no_particles):
                p_event_type.append({"type": field_types[j], "p": p_event[j], "incoming": False, "flavour": flavours[j]})
            me_sq = qed_matrix_element(p_event_type, constants['masses_psi'], constants['e'])

        obs_val = obs_fun(p_event)
        #print(f"Event {i+1}/{no_events}: |M|^2 = {me_sq}, weight = {weight_m}, observable = {obs_val}")
        ctheta_min=-0.95
        ctheta_max=0.95
        if ctheta_min <= obs_val <= ctheta_max:
            obs_values.append(obs_val)
            cross_section_values.append(me_sq * weight_m)# * obs_val)

    # Multiply summation by prefactor V/(F*N) to get <O>
    # V = weight_0
    weight_0 = ((np.pi / 2) ** (no_outgoing - 1)) * (com_energy ** (2 * no_outgoing - 4)) / (sp.gamma(no_outgoing) * sp.gamma(no_outgoing - 1))
    #<O> = V/(F*N) * sum(w_m * |M|**2 * O(phi))
    flux = mc.flux_factor(com_energy, masses_in)
    prefactor = weight_0 / (no_events * flux)
    
    # Bins
    bins = np.linspace(ctheta_min, ctheta_max, 41)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = np.diff(bins)

    # Histogram values
    cross_section_values = prefactor * np.array(cross_section_values)
    values_sum, _ = np.histogram(obs_values, bins=bins, weights=cross_section_values) # Here'weights' has new meaning purely enclosed in histogram function
    values_sum /= np.diff(bins)
    values_sq, _ = np.histogram(obs_values, bins=bins, weights=cross_section_values**2)
    yerrors = np.sqrt(values_sq) / np.diff(bins)

    # Plot histogram as bar chart
    plt.bar(bin_centers, values_sum, width=bin_widths, align='center', alpha=0.7, edgecolor='black')

    # Plot line connecting peaks
    plt.plot(bin_centers, values_sum, color='red', marker=None, linestyle='-', linewidth=2)

    # Plot errorbars on histogram
    plt.errorbar(bin_centers, values_sum, yerr=yerrors, fmt='none', ecolor='black', capsize=2, label='MC error')

    plt.xlabel("cosθ")
    plt.ylabel("dσ/d(cosθ)")
    
    plt.xlim([-1, 1])
    #plt.yscale("log")
    print(len(obs_values), no_events)
    return


if __name__ == "__main__":
    m_phi = 10
    masses_psi = {"electron": 0.000511, "muon": 0.1057, "tau": 1.7768}
    g = 1
    e = 1
    lambda_0 = 1
    constants = {'m_phi': m_phi, 'masses_psi': masses_psi, 'lambda_0': lambda_0, 'g': g, 'e': e}
    if False:
        print(differential_cross_section(1000, 100000, 4,
                                        masses = np.array([0, 0, 0, 0]),
                                        theory = "phi",
                                        constants = constants,
                                        obs_fun = obs.cos_theta))
        plt.show()
    if False:
        print(differential_cross_section(1000, 100000, 4,
                                        constants=constants,
                                        theory="yukawa",
                                        field_types=[YFT.PSI, YFT.PSIBAR, YFT.PHI, YFT.PHI],
                                        masses=np.array([masses_psi["electron"], masses_psi["electron"], m_phi, m_phi]),
                                        flavours=["electron", "electron", None, None],
                                        obs_fun=obs.cos_theta))
        plt.show()
    if False:
        print(differential_cross_section(1000, 100000, 4,
                                        constants=constants,
                                        theory="qed",
                                        field_types=[QFT.PSI, QFT.PSIBAR, QFT.PSI, QFT.PSIBAR],
                                        masses=np.array([masses_psi["electron"], masses_psi["electron"], masses_psi["electron"], masses_psi["electron"]]),
                                        flavours=["electron", "electron", "electron", "electron"],
                                        obs_fun=obs.cos_theta))
        plt.show()
    if True:
        n = 3
        particles = 2*n + 2
        print(differential_cross_section(1000, 1000, particles,
                                        constants=constants,
                                        theory="qed",
                                        field_types=[QFT.PSI, QFT.PSIBAR] * (n+1),
                                        masses=np.array([masses_psi["electron"], masses_psi["electron"]]*(n+1)),
                                        flavours=["electron", "electron"]*(n+1),
                                        obs_fun=obs.cos_theta_rand))
        plt.show()


#caption("Differential Cross Section dσ/dcosθ in 2 -> 2 Scattering"), n = 4, masses = None, "cosθ from angle between p1 and z-axis/p3"