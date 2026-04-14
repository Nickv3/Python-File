import numpy as np
from itertools import combinations, product
from enum import Enum, auto



def gamma_matrices():
    """
    Calculates the gamma matrices in the Dirac representation, used in slash operator and Dirac propagator.

    Returns:
    γ{0,1,2,3} (np.ndarray): gamma^mu matrices in the Dirac representation.
    """
    g0 = np.array([[ 1, 0, 0, 0],
                   [ 0, 1, 0, 0],
                   [ 0, 0,-1, 0],
                   [ 0, 0, 0,-1]], dtype=complex)

    g1 = np.array([[ 0, 0, 0, 1],
                   [ 0, 0, 1, 0],
                   [ 0,-1, 0, 0],
                   [-1, 0, 0, 0]], dtype=complex)

    g2 = np.array([[ 0, 0, 0,-1j],
                   [ 0, 0, 1j, 0],
                   [ 0, 1j, 0, 0],
                   [-1j, 0, 0, 0]], dtype=complex)

    g3 = np.array([[ 0, 0, 1, 0],
                   [ 0, 0, 0,-1],
                   [-1, 0, 0, 0],
                   [ 0, 1, 0, 0]], dtype=complex)

    return g0, g1, g2, g3

gamma0, gamma1, gamma2, gamma3 = gamma_matrices()

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


def photon_polarization(p, helicity):
    return np.array(p, dtype=complex)
def photon_polarization_0(p, helicity):
    px, py, pz = p[1:]
    k = np.array([px, py, pz])
    k_norm = np.linalg.norm(k)
    if k_norm == 0:
        raise ValueError("Photon momentum cannot be zero")
    k = k / k_norm

    # Arbitrary reference vector, z direction unless photon close to z-axis, then x to avoid 0 cross products.
    if abs(k[2]) < 0.9:
        ref = np.array([0,0,1])
    else:
        ref = np.array([1,0,0])

    # Transverse basis (e1, e2) orthogonal to k
    e1 = np.cross(k, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(k, e1)

    # Helicity states
    if helicity == +1:
        eps_spatial = (e1 + 1j * e2) / np.sqrt(2)
    elif helicity == -1:
        eps_spatial = (e1 - 1j * e2) / np.sqrt(2)
    else:
        raise ValueError("Helicity must be ±1")
    
    return np.array([0, *eps_spatial], dtype=complex)


def helicity_spinor(p, helicity):
    px, py, pz = p[1:]
    p_norm = np.linalg.norm([px, py, pz])
    if p_norm == 0:
        raise ValueError("Momentum cannot be zero")

    # angles
    theta = np.arccos(pz / p_norm)
    phi = np.arctan2(py, px)

    # Chi spinors for helicity states
    if helicity == +1: # chi+ = (cos(theta/2), exp(i*phi)*sin(theta/2))^T
        return np.array([[np.cos(theta/2)], [np.exp(1j*phi)*np.sin(theta/2)]], dtype=complex)
    elif helicity == -1: # chi- = (-sin(theta/2), exp(i*phi)*cos(theta/2))^T
        return np.array([[-np.sin(theta/2)], [np.exp(1j*phi)*np.cos(theta/2)]], dtype=complex)
    else:
        raise ValueError("Helicity must be ±1")

def dirac_spinor_u(p, m_psi, helicity):
    """
    Calculates the Dirac spinor u(p,s) for an external fermion current in the Dirac basis.

    Parameters:
    p (np.ndarray): four-momentum [E, px, py, pz], abs(E) input so crossing does not give complex spinors
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    helicity (int): helicity state (±1)

    Returns:
    u(p,s) (np.ndarray): external fermion current u in Dirac basis. Output shape (4,1).
    """
    E, px, py, pz = p
    chi = helicity_spinor(p, helicity)

    sigma_dot_p = np.array([
        [pz, px - 1j*py],
        [px + 1j*py, -pz]])
    
    eta = np.sqrt(E + m_psi)
    upper = eta * chi
    lower = (sigma_dot_p @ chi) / eta

    return np.vstack([upper, lower])

def dirac_spinor_v(p, m_psi, helicity):
    """
    Calculates the Dirac spinor v(p,s) for an outgoing antifermion in the Dirac basis. Later conjugate and transpose to get vbar = v†γ^0 for the external antifermion current.

    Params:
    p (np.ndarray): four-momentum [E, px, py, pz], abs(E) input so crossing does not give complex spinors
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    helicity (int): helicity state (±1)

    Returns:
    v(p,s) (np.ndarray): outgoing antifermion current v in Dirac basis. Output shape (4,1)
    """
    E, px, py, pz = p
    chi = helicity_spinor(p, -helicity)

    sigma_dot_p = np.array([
        [pz, px - 1j*py],
        [px + 1j*py, -pz]])

    eta = np.sqrt(E + m_psi)
    upper = (sigma_dot_p @ chi) / eta
    lower = eta * chi

    return np.vstack([upper, lower])

# FieldType class assigns names to indices of different fields.
class FieldType(Enum):
    A = auto()      # A_μ
    PSI = auto()    # ψ
    PSIBAR = auto() # ψ̄ 

class Current:
    def __init__(self, field_type, current, momentum):
        """
        Created a current object for each external particle/combined current, and assign parameters as its values.

        Parameters:
        self: current object to be created
        field_type (FieldType): type of the current (A, PSI, PSIBAR)
        current (np.ndarray or float): the current value ( (4,) 4-vector for A and (4,1)/(1,4) spinors/matrices for PSI/PSIBARs)
        momentum (np.ndarray): four-momentum [E, px, py, pz]

        Returns:
        Current object with specified type, current value, and momentum.
        """
        self.type = field_type
        self.current = current
        self.p = np.asarray(momentum, float)

    def propagate(self, m_psi, eps = 1e-12):
        """
        Creates a new current object by applying the appropriate propagator to the new combined current, based on its field type.
        Premultiplies vector for ψ̄ and postmultiply for ψ, and postmultiply for matrix A_μ.

        Parameters:
        self (Current): current object to be propagated
        m_psi (float): mass of the fermion field ψ/ψ̄ 
        eps (float): imaginary part added to denominator to avoid limits in propagator, can made smaller if causing issues

        Returns:
        Modified current object with new current value, but unchanged type and momentum.
        """
        p_sq = self.p[0]**2 - np.dot(self.p[1:], self.p[1:])

        # Photon propagator: D(p) = (-i g^{μν}) / (p^2 + iε)
        if self.type == FieldType.A:
            metric = np.diag([1, -1, -1, -1])
            D_prop = (-1j * metric) / (p_sq + 1j * eps) # eps here is a small imiginary part, NOT the photon polarisation vector
            return Current(self.type, D_prop @ self.current, self.p)

        # Dirac/vector/fermion propagator: S(p) = i (p-slash + m) / (p^2 - m^2 + iε)
        if self.type == FieldType.PSI:
            S_prop = 1j * (slash(self.p) + m_psi*np.eye(4)) / (p_sq - m_psi**2 + 1j * eps)
            return Current(self.type, self.current @ S_prop, self.p)
        if self.type == FieldType.PSIBAR:
            S_prop = 1j * (slash(self.p) + m_psi*np.eye(4)) / (p_sq - m_psi**2 + 1j * eps)
            return Current(self.type, S_prop @ self.current, self.p)

        raise ValueError("Unknown field type")

    def combine(self, other, e):
        """
        Creates a new current object by combining the current values of self and other, applying Feynman rules for a vertexbased on their field types.
        Will not combine if the vertex is not valid for the QED interaction (i.e. 2 of same type or non-existent type), and will raise an error instead.

        Parameters:
        self (Current): current object to be propagated
        other (Current): current object to be combined with self
        e (float): coupling constant for QED interaction

        Returns:
        Combined current object with new type, current value and momentum.
        """
        p_new = self.p + other.p

        #CHECK if - sign needed vertices in -ee_f(gamma^mu)_alpha_beta
        # ψ + ψ̄ → A_μ
        if self.type == FieldType.PSI and other.type == FieldType.PSIBAR:
            j_mu = np.zeros(4, dtype=complex)
            for mu, gamma in enumerate([gamma0, gamma1, gamma2, gamma3]):
                j_mu[mu] = (self.current @ gamma @ other.current)[0,0] # [0,0] extracts float from 1x1 array
            return Current(FieldType.A, 1j * e * j_mu, p_new)
        if self.type == FieldType.PSIBAR and other.type == FieldType.PSI:
            j_mu = np.zeros(4, dtype=complex)
            for mu, gamma in enumerate([gamma0, gamma1, gamma2, gamma3]):
                j_mu[mu] = (other.current @ gamma @ self.current)[0,0]
            return Current(FieldType.A, 1j * e * j_mu, p_new)

        # ψ + A_μ → ψ
        if self.type == FieldType.PSI and other.type == FieldType.A:
            slash_eps = sum(other.current[mu] * gamma 
                            for mu, gamma in enumerate([gamma0, gamma1, gamma2, gamma3]))
            return Current(FieldType.PSI, 1j * e * (self.current @ slash_eps), p_new)
        if self.type == FieldType.A and other.type == FieldType.PSI:
            slash_eps = sum(self.current[mu] * gamma 
                            for mu, gamma in enumerate([gamma0, gamma1, gamma2, gamma3]))
            return Current(FieldType.PSI, 1j * e * (other.current @ slash_eps), p_new)

        # ψ̄ + A_μ → ψ̄
        if self.type == FieldType.PSIBAR and other.type == FieldType.A:
            slash_eps = sum(other.current[mu] * gamma 
                            for mu, gamma in enumerate([gamma0, gamma1, gamma2, gamma3]))
            return Current(FieldType.PSIBAR, 1j * e * (slash_eps @ self.current), p_new)
        if self.type == FieldType.A and other.type == FieldType.PSIBAR:
            slash_eps = sum(self.current[mu] * gamma 
                            for mu, gamma in enumerate([gamma0, gamma1, gamma2, gamma3]))
            return Current(FieldType.PSIBAR, 1j * e * (slash_eps @ other.current), p_new)


        raise ValueError(f"Invalid QED vertex: {self.type} + {other.type}")



# External currents, classified by field type and contains the spinor current (or 1 for scalars) and momentum information for the external particles.
def external_current(field_type, p, m_psi, incoming, crossed, helicity=None):
    """
    Creates a current object for an external particle based on its field type, momentum, and spinor current (for fermions).

    Parameters:
    field_type (FieldType): type of the current (A, PSI, PSIBAR)
    p (np.ndarray): four-momentum [E, px, py, pz]
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    incoming (bool): whether the particle is incoming or outgoing, to determine which u/v spinor to construct
    helicity (int): helicity state (±1)

    Returns:
    Combined current object with new type, current value and momentum.
    """
    #print(f"field_type={field_type}, helicity={helicity}, incoming={incoming}")
    #print(crossed)
    if crossed == True: # Ensure crossed particles are calculated correctly, using energy component but new type and vector (helicity flipped globally).
        p_spinor = np.array([-p[0], p[1], p[2], p[3]])
    else:
        p_spinor = p

    if field_type == FieldType.A:
        eps = photon_polarization(p_spinor, helicity) # photon polarisation vector
        return Current(FieldType.A, eps, p)

    if field_type == FieldType.PSI:
        u = dirac_spinor_u(p_spinor, m_psi, helicity)
        if incoming == False:
            u = u.conj().T @ gamma0 # ubar = u†γ^0 is the external current for outgoing fermions (shape 1x4/row vector)
        return Current(FieldType.PSI, u, p)

    if field_type == FieldType.PSIBAR:
        v = dirac_spinor_v(p_spinor, m_psi, helicity)
        if incoming == True:
            v = v.conj().T @ gamma0 # vbar = v†γ^0 is the external current for incoming antifermions (shape 1x4/row vector)
        return Current(FieldType.PSIBAR, v, p)

    raise ValueError("Unknown field type")



def index_subsets(indices):
    """
    Splits set of indices to be combined into all unique pairs of non-empty subsets (a,b).

    Parameters:
    indices (tuple of ints): tuple of indices to be split into subsets

    Returns:
    subsets (list of tuples): list of unique subsets represented as a tuple of two tuples (a,b).
    """
    n = len(indices)
    subsets = []
    seen = set()

    for a_size in range(1, n // 2 + 1):
        for combo in combinations(indices, a_size):
            a = tuple(combo)
            b = tuple(i for i in indices if i not in combo)
            new = tuple(sorted((a, b)))
            if new not in seen:
                seen.add(new)
                subsets.append((a, b))
    return subsets



def calculate_amplitude(external_currents, m_psi, e):
    """
    Calculates the matrix element M for a given set of external particles with given helicity state using B-G recursion.

    Parameters:
    external_currents (list of Current objects): list of external currents for each particle
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    e (float): coupling constant for QED interaction

    Returns:
    M (float): complex M for the given external particles and helicity states
    """
    n = len(external_currents) - 1
    single_current = external_currents[0]
    other_currents = external_currents[1:]
    J = {}

    # Single-particle currents
    for i, pt in enumerate(other_currents):
        J[(i,)] = pt

    debug_contributions = {}

    # Build multi-particle currents
    for no_pt in range(2, n + 1): # no_pt is the number of particles in the current being constructed, starting from 2 up to n
        for inds in combinations(range(n), no_pt): # indices of the external particles included in the current being constructed
            total = None

            for a, b in index_subsets(inds):
                if a not in J or b not in J:
                    #print("fail1")
                    continue

                Ja = J[a]
                Jb = J[b]

                try:
                    C = Ja.combine(Jb, e)

                    key_d = inds
                    if key_d not in debug_contributions:
                        debug_contributions[key_d] = []
                    debug_contributions[key_d].append(C.current.copy())

                    # propagate only if not final off-shell current
                    if no_pt < n:
                        C = C.propagate(m_psi)

                    #total = C if total is None else Current(C.type, total.current + C.current, C.p) below without error handling
                    if total is None:
                        total = C
                    else:
                        if total.type != C.type:
                            raise ValueError("Type mismatch in current summation")
                        total = Current(C.type, total.current + C.current, C.p)

                except ValueError:
                    #print(f"fail2 for combination: {Ja.type} + {Jb.type}")
                    pass

            if total is not None:
                J[inds] = total

    # Final off-shell current (all outgoing combined)
    final_current = J[tuple(range(n))]
    if False: # Debug print all currents
        print("\n--- Checking for duplicate contributions in final current ---")

        final_key = tuple(range(n))

        if final_key in debug_contributions:
            contribs = debug_contributions[final_key]
            for i in range(len(contribs)):
                for j in range(i+1, len(contribs)):
                    if np.allclose(contribs[i], contribs[j]):
                        print(f"Duplicate contribution found: {i} and {j}")
                        pass


        for j in J:
            print(f"J{j} = {J[j].current}")

    #set up to contract n+1 particle off shell current with single particle current to get M
    if single_current.type == FieldType.PSI and final_current.type == FieldType.PSI: # single incoming u and final outgoing ubar
        M = (final_current.current @ single_current.current)[0,0]

    elif single_current.type == FieldType.PSIBAR and final_current.type == FieldType.PSIBAR: # single incoming vbar and final outgoing v
        M = (single_current.current @ final_current.current)[0,0]

    elif single_current.type == FieldType.A and final_current.type == FieldType.A: #incoming photon and outgoing photon
        metric = np.diag([1, -1, -1, -1])
        M = single_current.current @ metric @ final_current.current

    else:
        raise ValueError(f"Invalid final contraction structure {single_current.type} with {final_current.type}")
    print(f"M = {abs(M)}")
    return M

def helicity_averaged_matrix_element(external_points, m_psi, e):
    """
    Calculates the squared and helicity-averaged matrix element |M|^2 for a given set of external particles using B-G recursion.

    Constraints:
    Currently assumes only 2 helicity states (+1,-1).
    Currently assumes overall helicity and fermion (baryon/lepton?) number conservation.

    Parameters:
    external_points (list of dict): list of external particles with their type and momentum, e.g. [{"type": FieldType.PSI, "p": p0}, ...]
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    e (float): coupling constant for QED interaction

    Returns:
    M_sq_av (float): squared and helicity-averaged matrix element |M|^2 for the given external particles
    """
    # Determines the number of initial (anti-)fermions based on the external_points list, and generates all possible helicity configurations for them.
    initial_indices = [i for i, pt in enumerate(external_points) if pt["incoming"]] # Now all incoming and outgoing particles have a helicity as no scalars
    final_indices = [i for i, pt in enumerate(external_points)if not pt["incoming"]]
    #print(len(initial_fermion_indices))
    all_indices = initial_indices + final_indices
    helicity_configs = list(product([+1,-1], repeat=len(all_indices))) #Fermions and photons all have 2 helicity states
    #print(helicity_configs)

    crossed_points = []
    for i, pt in enumerate(external_points):
        new_pt = pt.copy() # avoid changing original
        p = new_pt["p"].copy()
        field_type = new_pt["type"]
        incoming = new_pt["incoming"]

        if incoming == True and i > 0: # crossing all but first incoming particle to have n+1 outgoing particles
            # Reverse 4-momentum.
            p = -p
            # Change field type (ψ ↔ ψ̄, φ is unchanged)
            if field_type == FieldType.PSI:
                field_type = FieldType.PSIBAR
            elif field_type == FieldType.PSIBAR:
                field_type = FieldType.PSI
            crossed_points.append({"p": p, "type": field_type, "incoming": False, "crossed": True})
            continue
        crossed_points.append({"p": p, "type": field_type, "incoming": incoming, "crossed": False})
        
    # Loop over spins of initial (anti-)fermions
    total_M_sq = 0.0
    for helicity_set in helicity_configs:
        #print("Running helicity config:", helicity)
        helicity_dict = {}
        for i, h in zip(all_indices, helicity_set):
            if crossed_points[i]["crossed"]:
                helicity_dict[i] = -h   # flip here
            else:
                helicity_dict[i] = h

        external_currents = []

        for i, pt in enumerate(crossed_points):
            current = external_current(pt["type"], pt["p"], m_psi, pt["incoming"], pt["crossed"], helicity=helicity_dict[i])
            external_currents.append(current)
        #print(external_currents)
        print(helicity_dict)
        M = calculate_amplitude(external_currents, m_psi, e)

        total_M_sq += abs(M)**2
    M_sq_av = total_M_sq / (2 ** len(initial_indices))
    return M_sq_av


if __name__ == "__main__":
    E = 1000
    m_psi = 10
    e = 1
    p = np.sqrt(E**2 - m_psi**2)

    p0 = np.array([E, 0, 0,  p])
    p1 = np.array([E, 0, 0, -p])
    print(p0)
    eps = photon_polarization(p0, helicity=+1)
    eps1 = photon_polarization(p0, helicity=-1)
    if False:
        None

    if True:
        theta = np.pi / 4
        p2 = np.array([E,  p*np.sin(theta), 0,  p*np.cos(theta)])
        p3 = np.array([E, -p*np.sin(theta), 0, -p*np.cos(theta)])
        momenta22 = (p0, p1, p2, p3)
        external_points22 = [
            {"type": FieldType.PSI,    "p": p0, "incoming": True},
            {"type": FieldType.PSIBAR, "p": p1, "incoming": True},
            {"type": FieldType.A,    "p": p2, "incoming": False},
            {"type": FieldType.A,    "p": p3, "incoming": False},
            ]
        print(helicity_averaged_matrix_element(external_points22, m_psi, e))

    if False:
        E_A = 2*E/3
        k = np.sqrt(E_A**2 - 0**2)
        p2 = np.array([E_A,  k,                 0, 0])
        p3 = np.array([E_A, -k/2,  np.sqrt(3)*k/2, 0])
        p4 = np.array([E_A, -k/2, -np.sqrt(3)*k/2, 0])
        momenta23 = (p0, p1, p2, p3, p4)
        external_points23 = [
            {"type": FieldType.PSI,    "p": p0, "incoming": True},
            {"type": FieldType.PSIBAR, "p": p1, "incoming": True},
            {"type": FieldType.A,    "p": p2, "incoming": False},
            {"type": FieldType.A,    "p": p3, "incoming": False},
            {"type": FieldType.A,    "p": p4, "incoming": False}
            ]
        print(helicity_averaged_matrix_element(external_points23, m_psi, e))

