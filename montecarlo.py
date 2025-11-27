def monte_carlo_cross_section(N_particles, sqrt_s, matrix_element_func, n_events):
    """
    Estimate the cross-section in scattering of N_particles massless particles using Monte Carlo integration.

    Parameters:
    N_particles (int): Number of particles involved in scattering.
    sqrt_s (float): COM energy.
    matrix_element_func (callable): Function to compute the squared matrix element from currents.py.
    n_events (int): Number of Monte Carlo events to generate.

    Returns:
    float: Estimated cross-section.
    """

    # Initialize phase space generator
    masses = [0] * (N_particles - 2)  # Assuming massless outgoing particles
    ps_generator = PhaseSpaceGenerator(N_particles, sqrt_s, masses=masses, com_output=True, algorithm='ramboflow')

    # Generate random numbers for phase space generation
    ndim = (N_particles - 2) * 4 + 2
    xrand = np.random.rand(n_events, ndim)

    # Generate outgoing momenta and weights
    all_ps, wts, _, _, _ = ps_generator(xrand)


    # Compute matrix elements and accumulate weighted sum
    total_weighted_me_sq = 0 # Running total of w_i * |M(p_i)|^2
    for i in range(n_events):
        #print(np.array(all_ps)[i])
        me_sq = matrix_element_func(np.array(all_ps)[i])
        #print(f"Matrix element squared for event {i}:", me_sq)
        total_weighted_me_sq += me_sq * np.array(wts)[i]

    # Average over events and multiply by flux factor
    flux = 2 * (sqrt_s ** 2)
    cross_section = total_weighted_me_sq / n_events / flux

    return cross_section


import numpy as np
from madflow.phasespace import PhaseSpaceGenerator
from currents import calculate_matrix_element
if True:
    for _ in range(1):
        print(_)
        print(monte_carlo_cross_section(5, 10000, calculate_matrix_element, 10))

