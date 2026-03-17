import numpy as np
from itertools import combinations, product
from enum import Enum, auto



def gamma_matrices():
    """
    Calculates the gamma matrices in the Dirac representation, used in slash operator and Dirac propagator.

    Returns:
    g{0,1,2,3} (np.ndarray): gamma^mu matrices in the Dirac representation.
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



def dirac_spinor_u(p, m_psi, spin):
    """
    Calculates the Dirac spinor u(p,s) for an external fermion current in the Dirac basis.

    Parameters:
    p (np.ndarray): four-momentum [E, px, py, pz]
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    spin (int): spin state (0 or 1)

    Returns:
    u(p,s) (np.ndarray): external fermion current u in Dirac basis. Output shape (4,1).
    """
    E, px, py, pz = p
    chi = np.array([[1],[0]]) if spin == 0 else np.array([[0],[1]])

    sigma_dot_p = np.array([
        [pz, px - 1j*py],
        [px + 1j*py, -pz]])
    
    eta = np.sqrt(E + m_psi) #abs as crossing can lead to negative energy in 4-momentum when we want positive for spinor construction
    upper = eta * chi
    lower = (sigma_dot_p @ chi) / eta

    return np.vstack([upper, lower])

def dirac_spinor_v(p, m_psi, spin):
    """
    Calculates the Dirac spinor v(p,s) for an outgoing antifermion in the Dirac basis. Later conjugate and transpose to get vbar = v†γ^0 for the external antifermion current.

    Params:
    p (np.ndarray): four-momentum [E, px, py, pz]
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    spin (int): spin state (0 or 1)

    Returns:
    v(p,s) (np.ndarray): outgoing antifermion current v in Dirac basis. Output shape (4,1)
    """
    E, px, py, pz = p
    chi = np.array([[1],[0]]) if spin == 0 else np.array([[0],[1]])

    sigma_dot_p = np.array([
        [pz, px - 1j*py],
        [px + 1j*py, -pz]])

    eta = np.sqrt(E + m_psi) #abs as crossing can lead to negative energy in 4-momentum when we want positive for spinor construction
    upper = (sigma_dot_p @ chi) / eta
    lower = eta * chi

    return np.vstack([upper, lower])


# FieldType class assigns names to indices of different fields.
class FieldType(Enum):
    PHI = auto()    # φ
    PSI = auto()    # ψ
    PSIBAR = auto() # ψ̄ 

class Current:
    def __init__(self, field_type, current, momentum):
        """
        Created a current object for each external particle/combined current, and assign parameters as its values.

        Parameters:
        self: current object to be created
        field_type (FieldType): type of the current (PHI, PSI, PSIBAR)
        current (np.ndarray or float): the current value (scalar for PHI, 4x1 matrix for PSI/PSIBAR)
        momentum (np.ndarray): four-momentum [E, px, py, pz]

        Returns:
        Current object with specified type, current value, and momentum.
        """
        self.type = field_type
        self.current = current          # scalar or (4x1 or 1x4) array depending on field type
        self.p = np.asarray(momentum, float)

    def propagate(self, m_phi, m_psi, eps = 1e-12):
        """
        Creates a new current object by applying the appropriate propagator to the new combined current, based on its field type.
        Premultiplies line vector for ψ̄ and postmultiply for ψ, and scalar multiplication for φ.

        Parameters:
        self (Current): current object to be propagated
        m_phi (float): mass of the scalar field φ
        m_psi (float): mass of the fermion field ψ/ψ̄ 
        eps (float): imaginary part added to denominator to avoid limits in propagator, can made smaller if causing issues

        Returns:
        Modified current object with new current value, but unchanged type and momentum.
        """
        p_sq = self.p[0]**2 - np.dot(self.p[1:], self.p[1:])

        # Scalar propagator: D(p) = i / (p^2 - m^2 + iε)
        if self.type == FieldType.PHI:
            D_prop = 1j / (p_sq - m_phi**2 + 1j * eps)
            return Current(self.type, D_prop * self.current, self.p)

        # Dirac/vector/fermion propagator: S(p) = i (p-slash + m) / (p^2 - m^2 + iε)
        if self.type == FieldType.PSI:
            S_prop = 1j * (slash(self.p) + m_psi*np.eye(4)) / (p_sq - m_psi**2 + 1j * eps)
            return Current(self.type, self.current @ S_prop, self.p)
        if self.type == FieldType.PSIBAR:
            S_prop = 1j * (slash(self.p) + m_psi*np.eye(4)) / (p_sq - m_psi**2 + 1j * eps)
            return Current(self.type, S_prop @ self.current, self.p)

        raise ValueError("Unknown field type")

    def combine(self, other, g):
        """
        Creates a new current object by combining the current values of self and other, applying Feynman rules for a vertexbased on their field types.
        Will not combine if the vertex is not valid for the Yukawa interaction (i.e. 2 of same type or non-existent type), and will raise an error instead.

        Parameters:
        self (Current): current object to be propagated
        other (Current): current object to be combined with self
        g (float): coupling constant for Yukawa interaction

        Returns:
        Combined current object with new type, current value and momentum.
        """
        p_new = self.p + other.p

        # ψ + ψ̄ → φ
        if self.type == FieldType.PSI and other.type == FieldType.PSIBAR:
            vertex = 1j * g * (self.current @ other.current)[0,0] # [0,0] extracts float from 1x1 array
            return Current(FieldType.PHI, vertex, p_new)
        if self.type == FieldType.PSIBAR and other.type == FieldType.PSI:
            vertex = 1j * g * (other.current @ self.current)[0,0]
            return Current(FieldType.PHI, vertex, p_new)

        # ψ + φ → ψ
        if self.type == FieldType.PSI and other.type == FieldType.PHI:
            return Current(FieldType.PSI, 1j * g * self.current * other.current, p_new)
        if self.type == FieldType.PHI and other.type == FieldType.PSI:
            return Current(FieldType.PSI, 1j * g * other.current * self.current, p_new)

        # ψ̄ + φ → ψ̄
        if self.type == FieldType.PSIBAR and other.type == FieldType.PHI:
            return Current(FieldType.PSIBAR, 1j * g * self.current * other.current, p_new)
        if self.type == FieldType.PHI and other.type == FieldType.PSIBAR:
            return Current(FieldType.PSIBAR, 1j * g * other.current * self.current, p_new)


        raise ValueError(f"Invalid Yukawa vertex: {self.type} + {other.type}")



# External currents, classified by field type and contains the spinor current (or 1 for scalars) and momentum information for the external particles.
def external_current(field_type, p, m_psi, incoming, crossed, spin=None):
    """
    Creates a current object for an external particle based on its field type, momentum, and spinor current (for fermions).

    Parameters:
    field_type (FieldType): type of the current (PHI, PSI, PSIBAR)
    p (np.ndarray): four-momentum [E, px, py, pz]
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    incoming (bool): whether the particle is incoming or outgoing, to determine which u/v spinor to construct
    spin (int): spin state (0 or 1)

    Returns:
    Combined current object with new type, current value and momentum.
    """
    #print(f"field_type={field_type}, spin={spin}, incoming={incoming}")
    #print(crossed)
    if crossed == True: # Ensure crossed particles are calculated correctly, using original momentum but new type.
        p_spinor = -p
    else:
        p_spinor = p

    if field_type == FieldType.PHI:
        return Current(FieldType.PHI, 1.0 + 0j, p)

    if field_type == FieldType.PSI:
        u = dirac_spinor_u(p_spinor, m_psi, spin)
        if incoming == False:
            u = u.conj().T @ gamma0 # ubar = u†γ^0 is the external current for outgoing fermions (shape 1x4/row vector)
        return Current(FieldType.PSI, u, p)

    if field_type == FieldType.PSIBAR:
        v = dirac_spinor_v(p_spinor, m_psi, spin)
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



def calculate_amplitude(external_currents, m_phi, m_psi, g):
    """
    Calculates the matrix element M for a given set of external particles with given spin state using B-G recursion.

    Parameters:
    external_currents (list of Current objects): list of external currents for each particle
    m_phi (float): mass of the scalar field φ
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    g (float): coupling constant for Yukawa interaction

    Returns:
    M (float): complex M for the given external particles and spin states
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
                    C = Ja.combine(Jb, g)

                    key_d = inds
                    if key_d not in debug_contributions:
                        debug_contributions[key_d] = []
                    debug_contributions[key_d].append(C.current.copy())

                    # propagate only if not final off-shell current
                    if no_pt < n:
                        C = C.propagate(m_phi, m_psi)

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

    elif single_current.type == FieldType.PHI and final_current.type == FieldType.PHI: #incoming scalar and outgoing scalar
        M = single_current.current * final_current.current

    else:
        raise ValueError(f"Invalid final contraction structure {single_current.type} with {final_current.type}")
    #print(f"M = {M}")
    return M

def spin_averaged_matrix_element(external_points, m_phi, m_psi, g):
    """
    Calculates the squared and spin-summed matrix element |M|^2 for a given set of external particles using B-G recursion.

    Changes to be made:
    Currently assumes all particles are outgoing, and the first two particles in the external_points list are the initial state particles (the 2nd of which will be conjugated for the current construction).
    Currently assumes only 2 spin states (0/up and 1/down).
    Currently assumes all particles are distinguishable, so no symmetry factors are included.
    Currently assumes overall spin and fermion (baryon/lepton?) number conservation.

    Parameters:
    external_points (list of dict): list of external particles with their type and momentum, e.g. [{"type": FieldType.PSI, "p": p0}, ...]
    m_phi (float): mass of the scalar field φ
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    g (float): coupling constant for Yukawa interaction

    Returns:
    M_sq_av (float): squared and spin-summed matrix element |M|^2 for the given external particles
    """
    # Determines the number of initial (anti-)fermions based on the external_points list, and generates all possible spin configurations for them.
    initial_fermion_indices = [i for i, pt in enumerate(external_points) if (pt["type"] in (FieldType.PSI, FieldType.PSIBAR)) and (pt["incoming"] == True)]
    #print(len(initial_fermion_indices))
    spin_configs = list(product([0,1], repeat=len(initial_fermion_indices)))
    #print(spin_configs)

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
            incoming = False
            crossed = True
            crossed_points.append({"p": p, "type": field_type, "incoming": incoming, "crossed": crossed})
            continue
        crossed_points.append({"p": p, "type": field_type, "incoming": incoming, "crossed": False})
        
    # Loop over spins of initial (anti-)fermions
    total_M_sq = 0.0
    for spins in spin_configs:
        #print("Running spin config:", spins)
        spin_dict = dict(zip(initial_fermion_indices, spins))

        external_currents = []

        for i, pt in enumerate(crossed_points):
            if i in spin_dict:
                current = external_current(pt["type"], pt["p"], m_psi, pt["incoming"], pt["crossed"], spin=spin_dict[i])
            else:
                current = external_current(pt["type"], pt["p"], m_psi, pt["incoming"], pt["crossed"])

            external_currents.append(current)
        #print(external_currents)
        M = calculate_amplitude(external_currents, m_phi, m_psi, g)

        total_M_sq += abs(M)**2
    M_sq_av = total_M_sq / (2 ** len(initial_fermion_indices))
    return M_sq_av


if __name__ == "__main__":
    E = 1000
    m_phi = 10
    m_psi = 10
    g = 1
    p = np.sqrt(E**2 - m_psi**2)

    p0 = np.array([E, 0, 0,  p])
    p1 = np.array([E, 0, 0, -p])
    if True:
        theta = np.pi / 4
        p2 = np.array([E,  p*np.sin(theta), 0,  p*np.cos(theta)])
        p3 = np.array([E, -p*np.sin(theta), 0, -p*np.cos(theta)])
        momenta22 = (p0, p1, p2, p3)
        external_points22 = [
            {"type": FieldType.PSI,    "p": p0, "incoming": True},
            {"type": FieldType.PSIBAR, "p": p1, "incoming": True},
            {"type": FieldType.PHI,    "p": p2, "incoming": False},
            {"type": FieldType.PHI,    "p": p3, "incoming": False},
            ]
        print(spin_averaged_matrix_element(external_points22, m_phi, m_psi, g))

    if True:
        E_phi = 2*E/3
        k = np.sqrt(E_phi**2 - m_phi**2)
        p2 = np.array([E_phi,  k,                 0, 0])
        p3 = np.array([E_phi, -k/2,  np.sqrt(3)*k/2, 0])
        p4 = np.array([E_phi, -k/2, -np.sqrt(3)*k/2, 0])
        momenta23 = (p0, p1, p2, p3, p4)
        external_points23 = [
            {"type": FieldType.PSI,    "p": p0, "incoming": True},
            {"type": FieldType.PSIBAR, "p": p1, "incoming": True},
            {"type": FieldType.PHI,    "p": p2, "incoming": False},
            {"type": FieldType.PHI,    "p": p3, "incoming": False},
            {"type": FieldType.PHI,    "p": p4, "incoming": False}
            ]
        print(spin_averaged_matrix_element(external_points23, m_phi, m_psi, g))