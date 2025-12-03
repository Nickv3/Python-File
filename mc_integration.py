import numpy as np
import scipy.special as sp
from currents import calculate_matrix_element

def mc_cross_section(com_energy, no_events, no_particles):
    """
    Estimate the cross-section in scattering of N_particles massless particles using Monte Carlo integration and rambo phase space generator.

    Parameters:
    com_energy (float): COM energy.
    no_events (int): Number of Monte Carlo events to generate.
    no_particles (int): Number of particles involved in scattering.

    Returns:
    cross_section (float): Estimated cross-section.
    """
    total_weighted_me_sq = 0 # Running total of w_i * |M(p_i)|^2

    phase_space = np.random.rand(no_events, no_particles, 4)
    weights = np.zeros(no_events)
    for i in range(no_events):
        phase_space[i], weights[i] = generate_phase_space(com_energy, phase_space[i])
    for i in range(no_events):
        me_sq = calculate_matrix_element(phase_space[i])
        print(f"Event {i+1}/{no_events}: |M|^2 = {me_sq}, weight = {weights[i]}")
        total_weighted_me_sq += me_sq * weights[i]

    # Average over events and multiply by flux factor
    flux = 2 * (com_energy ** 2)
    cross_section = total_weighted_me_sq / no_events / flux

    return cross_section

def generate_phase_space(w, particles_rhos):
    """
    Use rambo algorithm to generate n-particle phase space for massless particles.

    Parameters:
    w (float): COM energy.
    particles_rhos (np.ndarray): Random numbers for phase space generation. 2D array of shape (n_particles, 4).

    Returns:
    p_list (np.ndarray): Generated 4-momenta of n particles. 2D array of shape (n_particles, 4).
    """
    n = len(particles_rhos)
    q_list = np.zeros((n, 4))
    #Loop over each particle's random numbers to generate its four-momentum, q**2 = 0 for most particles so constrains masslessness
    for i in range(n):
        (rho_1, rho_2, rho_3, rho_4) = particles_rhos[i]
        c = 2 * rho_1 - 1
        phi = 2 * np.pi * rho_2
        q_0 = - np.log(rho_3 * rho_4)
        q_1 = q_0 * np.sqrt(1 - c**2) * np.cos(phi)
        q_2 = q_0 * np.sqrt(1 - c**2) * np.sin(phi)
        q_3 = q_0 * c
        q_list[i] = np.array([q_0, q_1, q_2, q_3])
    
    #Calculate total four-momentum and other useful quantities for transformation
    (Q_0, Q_1, Q_2, Q_3) = sum(q_list)
    M = np.sqrt(Q_0**2 - Q_1**2 - Q_2**2 - Q_3**2)
    gamma = Q_0 / M
    x = w/M
    a = 1/(1 + gamma)
    b_vector = - np.array([Q_1, Q_2, Q_3]) / M

    #Boost and scale each particle's four-momentum
    p_list = q_list.copy()
    for i in range(len(q_list)):
        (q_0, q_1, q_2, q_3) = q_list[i]
        q_vector = np.array([q_1, q_2, q_3])
        b_dot_q = sum(b_vector * q_vector)
        p_0 = x * (gamma * q_0 + b_dot_q)
        (p_1, p_2, p_3) = x * (q_vector + b_vector * q_0 + a * b_dot_q * b_vector)

        p_list[i] = np.array([p_0, p_1, p_2, p_3])
    #Calculate weight for this phase space point
    weight = ((np.pi / 2) ** (n - 1)) * ((w ** (2 * n - 4)) / sp.gamma(n) / sp.gamma(n - 1))
    
    #print(q_list)
    #print(sum(q_list))
    #print(p_list)
    #print(sum(p_list))
    return p_list, weight

if __name__ == "__main__":
    print(mc_cross_section(100000,10000,6))
