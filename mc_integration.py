import numpy as np
import math
import scipy.special as sp
from scipy.optimize import brentq
from currents import calculate_matrix_element

def mc_cross_section(com_energy, no_events, no_particles, masses):
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

    weight_0 = ((np.pi / 2) ** (no_particles - 1)) * ((com_energy ** (2 * no_particles - 4)) / sp.gamma(no_particles) / sp.gamma(no_particles - 1))
    for i in range(no_events):
        phase_space, weight_m = generate_phase_space(com_energy, no_particles, masses)
        me_sq = calculate_matrix_element(phase_space)
        weight = weight_m * weight_0
        print(f"Event {i+1}/{no_events}: |M|^2 = {me_sq}, weight = {weight}")
        total_weighted_me_sq += me_sq * weight

    # Average over events and multiply by flux factor
    flux = 2 * (com_energy ** 2)
    cross_section = total_weighted_me_sq / no_events / flux

    return cross_section


def generate_phase_space(w, n, masses):
    """
    Uses rambo algorithm to generate n-particle phase space for massless particles.
    When masses provided, transforms momenta to massive particles and provide weights.

    Parameters:
    w (float): COM energy.
    n (int): Number of particles involved in scattering.
    masses (np.ndarray): 1D array of particle weights. If None, massless particles calculated.

    Returns:
    p_list (np.ndarray): Generated 4-momenta of n particles for event. 2D array of shape (n, 4).
    weight (float): Weight value for event.
    """
    p_list = massless_momenta_generation(w, n)
    #Calculate weight for this phase space point
    if masses is not None:
        if len(masses) == n:
            print("Massive case")
            p_list, weight = massive_transform(w, n, p_list, masses)
        else:
            weight = 1
            raise IndexError("Incorrect # of masses")
    else:
        weight = 1
        print("Massless case")
    return p_list, weight

def massless_momenta_generation(w, n):
    """
    Use rambo algorithm to generate n-particle phase space for massless particles.
    Where (q or p)**2 = 0 shows masslessness of generated particle momenta.

    Parameters:
    w (float): COM energy.
    n (int): Number of particles involved in scattering.

    Returns:
    p_list (np.ndarray): Generated 4-momenta of n massless particles. 2D array of shape (n_particles, 4).
    """
    # Vectorised random number generation of n untransformed 4-momenta, q
    rho_1, rho_2, rho_3, rho_4 = np.random.rand(4, n) # Each rho_i of length n
    c = 2 * rho_1 - 1
    s = np.sqrt(1 - c**2)
    phi = 2 * np.pi * rho_2
    q_0 = - np.log(rho_3 * rho_4)
    q_list = np.empty((n, 4)) # Empty list of particle momenta
    q_list[:, 0] = q_0
    q_list[:, 1] = q_0 * s * np.cos(phi)
    q_list[:, 2] = q_0 * s * np.sin(phi)
    q_list[:, 3] = q_0 * c

    #Calculate total four-momentum and other useful quantities for transformation
    Q_0, Q_1, Q_2, Q_3 = np.sum(q_list, axis=0)
    M = np.sqrt(Q_0**2 - Q_1**2 - Q_2**2 - Q_3**2)
    gamma = Q_0 / M
    x = w/M
    a = 1/(1 + gamma)
    b_v = - np.array([Q_1, Q_2, Q_3]) / M

    # Vectorised boosting and scaling n 4-momenta, to transform p -> q
    q_v = q_list[:, 1:4] 
    b_dot_q = q_v @ b_v
    p_0 = x * (gamma * q_0 + b_dot_q)
    p_v = x * (q_v + b_v * q_0[:, None] + a * b_dot_q[:, None] * b_v)
    p_list = np.column_stack([p_0, p_v])
    #print(p_list)
    #print(sum(p_list))
    return p_list

def massive_transform(w, n, p_list, masses):
    """
    Use rambo algorithm to transform n-massless-particle phase space for massive particles.
    Where (q or p)**2 = 0 shows masslessness of generated particle momenta.

    Parameters:
    w (float): COM energy.
    n (int): Number of particles involved in scattering.
    p_list (np.ndarray): 2D array of generated momenta.
    masses (np.ndarray): 1D array of particle weights. If None, massless particles calculated.

    Returns:
    k_list (np.ndarray): Generated 4-momenta of n massive particles for event. 2D array of shape (n, 4).
    weight (float): Weight value for event.
    """
    # Solve w = Σ sqrt(m_i^2 + ξ^2 p_i^2) for ξ ≥ 0 numerically
    p_0 = p_list[:, 0] 
    p_v = p_list[:, 1:4]
    xi = solve_for_xi(w, p_0, masses)

    # Transform massless to massive momenta
    k_0 = np.sqrt(masses**2 + (xi * p_0)**2)
    k_v = xi * p_v
    k_list = np.column_stack([k_0, k_v])

    # Calculate event weight
    mag_k_v = np.linalg.norm(k_v, axis=1)
    term1 = sum(mag_k_v)
    term2 = math.prod(mag_k_v/k_0)
    term3 = sum(mag_k_v**2/k_0)
    weight_m = (term1/w)**(2*n-3) * (term2/term3)

    return k_list, weight_m

def equation_xi(xi, w, p, m):
    """
    Equation to solve for ξ

    Params:
    xi (float): variable to solve for
    w (float): target value
    p (np.array): array of momenta p_i
    m (np.array): array of masses m_i

    Returns:
    f(ξ) (float): f(ξ) = Σ sqrt(m_i^2 + ξ^2 (p_i^0)^2) - w,
    """
    return np.sum(np.sqrt(m**2 + (xi**2)*(p**2))) - w


def solve_for_xi(w, p, m):
    """
    Solves w = Σ sqrt(m_i^2 + ξ^2(p_i^0)^2) for ξ ≥ 0.
    Returns error for runtime or if no solution exists.
    
    Params:
    w (float): target value
    p (np.array): array of momenta p_i
    m (np.array): array of masses m_i
    
    Returns:
    xi_solution (float): solution for ξ ≥ 0.
    """
    # Initially xi = 0, gives w = Σ m_i, tests if solution possible.
    xi_min = 0.0
    f_min = equation_xi(xi_min, w, p, m)

    if f_min > 0:
        raise ValueError("No solution: w is smaller than Σ m_i")

    # Test if solution exists for large xi. Increase reject condition later if often being rejected with large w. 
    xi_max = 1.0
    while equation_xi(xi_max, w, p, m) < 0:
        xi_max *= 2
        if xi_max > 1e12:
            raise RuntimeError("Failed to find root: w too large")

    # Solve equation
    xi_solution = brentq(equation_xi, xi_min, xi_max, args=(w, p, m))
    return xi_solution


if __name__ == "__main__":
    print(mc_cross_section(100000,100,6, masses = np.array([1,2,3,4,5,6])))
