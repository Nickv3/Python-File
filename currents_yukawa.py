from itertools import combinations      # For generating index combinations for particle currents
import numpy as np


# Gamma matrices, used in slash operator and Dirac propagator
def gamma_matrices():
    """
    Return:
    g{0,1,2,3} (np.ndarray): gamma^mu matrices in the Dirac representation.
    """
    g0 = np.array([
        [ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  0,  0, -1]], dtype=complex)

    g1 = np.array([
        [ 0,  0,  0,  1],
        [ 0,  0,  1,  0],
        [ 0, -1,  0,  0],
        [-1,  0,  0,  0]], dtype=complex)

    g2 = np.array([
        [ 0,  0,  0, -1j],
        [ 0,  0,  1j,  0],
        [ 0,  1j,  0,  0],
        [-1j,  0,  0,  0]], dtype=complex)

    g3 = np.array([
        [ 0,  0,  1,  0],
        [ 0,  0,  0, -1],
        [-1,  0,  0,  0],
        [ 0,  1,  0,  0]], dtype=complex)

    return g0, g1, g2, g3

gamma0, gamma1, gamma2, gamma3 = gamma_matrices()
print(type(gamma0), gamma0.shape)
# Slash operator for p-slash in Dirac propagator
def slash(p):
    """
    Computes p-slash = gamma^mu p_mu.

    Parameters:
    p (np.ndarray): four-momentum [E, px, py, pz]

    Returns:
    np.ndarray: 4x4 complex matrix representing p-slash
    """
    E, px, py, pz = p
    return (gamma0 * E - gamma1 * px - gamma2 * py - gamma3 * pz)


def calculate_matrix_element(external_momenta, masses, m_phi = 0.1, m_psi = 0.1, g = 0.01):
    """
    Compute the final matrix element |M|^2 for a set of external points.

    Parameters:
    external_points: 2D list [[E, px, py, pz], ...]
    masses (np.nd.array): 1D array of masses of n particles.

    Returns:
    float: |M|^2
    """
    if len(external_momenta) < 2:
        return 0  # No interaction with fewer than 2 particles

    combined_current = calculate_combined_current(tuple(external_momenta[1:]), masses, m_phi, m_psi, g)
    matrix_element_squared = abs(combined_current)**2 #change to new form
    return matrix_element_squared



def calculate_combined_current(external_momenta, masses, m_phi, m_psi, g):
    """
    Calculate the combined currents for a set of external points.
    J_{12...n} = sum over all combinations of sub-currents with propagators and coupling constants.

    Parameters:
    external_momenta (np.ndarray): ((E, px, py, pz), ...], with first point removed (may no longer be true)
    g (float): coupling constant
    masses (np.nd.array): 1D array of masses of n particles.

    Returns:
    float: J_{12...n}
    """
    n = len(external_momenta)
    m_sq = 0 # Baseline squared mass set as zero and only changed later if masses given

    J_values = {}

    # Single particle currents
    J_1 = { (i,): 1 for i in range(n) }
    J_values["J_1"] = J_1

    # Build higher level particle currents, where 'dim' is the total number of combined external points.
    for no_points in range(2, n + 1):
        J_no_points = {}                                  # Initialise dictionary of currents for 'no_points' external points
        for current_indices in combinations(range(n), no_points):                        # Iterate over all index combinations for n-particle current
            points_subset = tuple(external_momenta[i] for i in current_indices)               # Get the external momenta for the current index combination
            pi_splits = index_subsets(current_indices, no_points)                  # Find all possible current combinations that can form the n-particle current
            
            current_sum = 0
            for (pi_1, pi_2) in pi_splits:                                               # Iterate over all current combinations
                current_sum += J_values[f"J_{len(pi_1)}"][pi_1]*J_values[f"J_{len(pi_2)}"][pi_2]    # Sum the product of the two sub-currents
            
            if current_type == 'psi':
                propagator = calculate_dirac_propagator(points_subset, m_psi)
            elif current_type == 'phi':
                propagator = calculate_scalar_propagator(points_subset, m_phi)
            if no_points < n:
                propagator = 1

            J_no_points[current_indices] = propagator * (1j * g) * current_sum           # Calculate the n-particle current value

        J_values[f"J_{no_points}"] = J_no_points

    return J_values[f"J_{n}"][tuple(range(n))]



def index_subsets(pi:tuple, n:int):
    """
    Calculate all unique ways to split a set of indices into two non-empty subsets.
    
    Parameters:
    pi (tuple): Tuple of indices to split
    n (int): Number of indices in pi

    Returns:
    List of tuples of tuple pairs for pi_1 & pi_2
    """
    pi_pairs = []
    seen = set()
    pi_list = list(pi)

    def canonical(a, b):
        # Size ordering
        if len(a) < len(b):
            return (a, b)
        elif len(a) > len(b):
            return (b, a)
        # Equal size -> number ordering
        return (a, b) if a <= b else (b, a)
    
    #Subset sizes only up to n//2 avoids duplicates
    for r in range(1, n // 2 + 1):
        for combo in combinations(pi_list, r):
            left = tuple(combo)
            right = tuple(x for x in pi_list if x not in combo)
            
            #Sort pi_1 and pi_2 so A|B == B|A
            pair = canonical(left, right)

            # Must deduplicate using a hashable key
            if pair not in seen:
                seen.add(pair)
                pi_pairs.append(pair)

    return pi_pairs



def calculate_scalar_propagator(combined_external_points, m_phi):
    """
    Calculate the scalar propagator factor for a combined fermion and anti-fermion.
    
    Parameters:
    combined_external_points (tuple of arrays) ([E, px, py, pz], ...)
    m_phi (float): intermediate scalar propagator mass.

    Returns:
    Complex propagator factor
    """
    E, px, py, pz = sum(combined_external_points)
    p_sq = E*E - (px*px + py*py + pz*pz)
    return 1j / (p_sq - m_phi**2)

def calculate_dirac_propagator(combined_external_points, m_psi):
    """
    Calculate the (anti-)fermion propagator factor for a combined scalar and (anti-)fermion.
    
    Parameters:
    combined_external_points: tuple of lists/arrays ([E, px, py, pz], ...)
    m_psi (float): intermediate dirac propagator mass.

    Returns:
    Vector propagator factor
    """
    E, px, py, pz = sum(combined_external_points)
    p_sq = E*E - (px*px + py*py + pz*pz)



    return 1j / (p_sq - m_psi**2)


if __name__ == '__main__' and True:
    import numpy as np
    seed = 42
    #Define phase space points
    E = 100
    p = 10
    theta_list = [np.pi/4] #np.linspace(0, np.pi, 100)
    for theta in theta_list:
        p0 = np.array([E, 0, 0, p])
        p1 = -np.array([E, 0, 0, -p])
        p2 = np.array([E, p*np.sin(theta), 0, p*np.cos(theta)])
        p3 = np.array([E, -p*np.sin(theta), 0, -p*np.cos(theta)])
        #p4 = np.array([E, 0, p, 0])
        p_list = [p0, p1, p2, p3]
        #print(p_list)

        #Run functions
        print(calculate_matrix_element(p_list))