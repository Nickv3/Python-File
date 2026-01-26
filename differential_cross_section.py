import mc_integration as mc
import observables as obs

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

def differential_cross_section(com_energy:float, no_events:int, no_particles:int, masses, obs_fun):
    """
    Produces a differential cross-section graph by observable 'obs_fun' in scattering of no_particles massless particles using Monte Carlo integration and rambo phase space generator.

    Parameters:
    com_energy (float): COM energy.
    no_events (int): Number of Monte Carlo events to generate.
    no_particles (int): Number of particles involved in scattering.
    masses (np.nd.array): 1D array of masses of n particles.
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
        # incoming momenta (2 particles)
        p_in = mc.incoming_momenta(com_energy, masses_in)

        # outgoing momenta (n particles), i.e. the phase space points
        p_out, weight_m = mc.generate_phase_space(com_energy, no_outgoing, masses_out)

        # full event momenta: [p1, p2, p3, ..., p_{n+2}]
        p_event = np.vstack((*p_in, p_out))
        #if i < 10:
        #    print(p_event)

        me_sq = mc.calculate_matrix_element(p_event)
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



print(differential_cross_section(1000, 200000, 4, masses = np.array([50,100,50,100]), obs_fun = obs.cos_theta)) #np.array([1,2,3,4])
plt.show()


#caption("Differential Cross Section dσ/dcosθ in 2 -> 2 Scattering"), n = 4, masses = None, "cosθ from angle between p1 and z-axis/p3"