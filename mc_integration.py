import numpy as np
import math
import scipy.special as sp
from scipy.optimize import brentq
from currents_phi import calculate_matrix_element
from currents_yukawa import FieldType as YFT, spin_averaged_matrix_element as yukawa_matrix_element
from currents_qed_v2 import FieldType as QFT, spin_averaged_matrix_element as qed_matrix_element
import observables as obs

def incoming_momenta(com_energy, masses_in):
    """
    Adds incoming momenta in the partonic CM frame to the generated final momenta, for 2->n scattering.

    Parameters:
    com_energy (float): COM energy.
    masses (np.nd.array): 1D array of masses of n particles.

    Returns:
    p1, p2 (np.ndarrays): incoming momenta defined by CM energy and 2 particles masses.
    """
    s = com_energy**2

    if masses_in is None:
        #  Massless case
        E1 = E2 = com_energy / 2
        p = E1
    else:
        # Massive case
        m1, m2 = masses_in[0], masses_in[1]
        p = np.sqrt((s - (m1 + m2)**2) * (s - (m1 - m2)**2)) / (2 * com_energy)
        E1 = (s + m1**2 - m2**2) / (2 * com_energy)
        E2 = (s + m2**2 - m1**2) / (2 * com_energy)

    p1 = np.array([E1, 0.0, 0.0, +p])
    p2 = np.array([E2, 0.0, 0.0, -p])
    return p1, p2


def flux_factor(com_energy, masses_in):
    """
    Calculate the flux factor for 2->n scattering. for massive and massless case.
    """
    s = com_energy**2
    if masses_in is None:
        return 2 * s
    m1, m2 = masses_in[0], masses_in[1]
    return 2 * np.sqrt((s - (m1+m2)**2)*(s - (m1-m2)**2))

def separate_in_and_out(no_particles, masses):
    """
    Separate incoming and outgoing particle masses and numbers.
    """
    no_outgoing = no_particles - 2
    if masses is not None:
        masses_in = masses[0:2]
        masses_out = masses[2:]
        assert len(masses_out) == no_outgoing, "Incorrect no of masses for outgoing particles"
    else:
        masses_in = None
        masses_out = None
    return masses_in, masses_out, no_outgoing

def mc_cross_section(com_energy:float, no_events:int, no_particles:int, constants, theory, field_types = None, masses = None, flavours = None, obs_fun = None):
    """
    Estimate the cross-section in scattering of N_particles massless particles using Monte Carlo integration and rambo phase space generator.

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
    
    # Separate incoming and outgoing particle masses and numbers
    masses_in, masses_out, no_outgoing = separate_in_and_out(no_particles, masses)

    # Calculate weighted sum of matrix element squared * observable value
    for i in range(no_events):
        # incoming momenta (2 particles)
        p_in = incoming_momenta(com_energy, masses_in)

        # outgoing momenta (n particles), i.e. the phase space points
        p_out, weight_m = generate_phase_space(com_energy, no_outgoing, masses_out)

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

        total_weighted_me_sq += me_sq * weight_m * obs_val

    # Multiply summation by prefactor V/(F*N) to get <O>
    # V = weight_0
    weight_0 = ((np.pi / 2) ** (no_outgoing - 1)) * (com_energy ** (2 * no_outgoing - 4)) / (sp.gamma(no_outgoing) * sp.gamma(no_outgoing - 1))

    #<O> = V/(F*N) * sum(w_m * |M|**2 * O(phi))
    flux = flux_factor(com_energy, masses_in)
    observable_mean = total_weighted_me_sq * weight_0 / (no_events * flux)

    return observable_mean


def generate_phase_space(w, n, masses):
    """
    Uses rambo algorithm to generate n-particle phase space for outgoingparticles.
    When masses provided, transforms momenta to massive particles and provide weights.

    Parameters:
    w (float): COM energy.
    n (int): Number of outgoing scattered particles.
    masses (np.ndarray): 1D array of outgoing particle weights. If None, massless particles calculated.

    Returns:
    p_list (np.ndarray): Generated 4-momenta of n outgoing particles for event. 2D array of shape (n, 4).
    weight (float): Weight value for event.
    """
    p_list = massless_momenta_generation(w, n)
    #Calculate weight for this phase space point
    if masses is not None:
        assert len(masses) == n, "Incorrect no of masses for outgoing particles"
        p_list, weight = massive_transform(w, n, p_list, masses)
    else:
        weight = 1
    return p_list, weight


def massless_momenta_generation(w, n):
    """
    Use rambo algorithm to generate n-particle phase space for massless particles.
    Where (q or p)**2 = 0 shows masslessness of generated particle momenta.

    Parameters:
    w (float): COM energy.
    n (int): Number of outgoing scattered particles.

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
    n (int): Number of outgoing scattered particles.
    p_list (np.ndarray): 2D array of generated outgoing momenta for massless particles.
    masses (np.ndarray): 1D array of outgoing particle weights. If None, massless particles calculated.

    Returns:
    k_list (np.ndarray): Generated 4-momenta of n outgoing massive particles for event. 2D array of shape (n, 4).
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
    m_phi = 10
    masses_psi = {"electron": 0.000511, "muon": 0.1057, "tau": 1.7768}
    g = 1
    lambda_0 = 1
    constants = {'m_phi': m_phi, 'masses_psi': masses_psi, 'lambda_0': lambda_0, 'g': g}
    print(mc_cross_section(1000, 100000, 4,
                           constants=constants,
                           theory="phi",
                           field_types=None,
                           masses=None,
                           flavours=None,
                           obs_fun=obs.cross_section))
    print(mc_cross_section(1000, 100000, 4,
                           constants=constants,
                           theory="yukawa",
                           field_types=[YFT.PSI, YFT.PSIBAR, YFT.PHI, YFT.PHI],
                           masses=np.array([masses_psi["electron"], masses_psi["electron"], m_phi, m_phi]),
                           flavours=["electron", "electron", None, None],
                           obs_fun=obs.cross_section))
