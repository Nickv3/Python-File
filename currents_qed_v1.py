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
gammas = [gamma0, gamma1, gamma2, gamma3] # gamma^mu
metric = np.diag([1., -1., -1., -1.])   # g^{mu nu}

def slash(p):
    """
    Computes p-slash = gamma^mu p_mu.

    Parameters:
    p (np.ndarray): four-momentum [E, px, py, pz]

    Returns:
    np.ndarray: 4x4 complex matrix representing p-slash
    """
    E, px, py, pz = p
    return gamma0 * E - gamma1 * px - gamma2 * py - gamma3 * pz



def photon_polarization_0(p, spin):
    return np.array(p, dtype=complex)
def photon_polarization(p, spin):
    px, py, pz = p[1], p[2], p[3]
    k = np.array([px, py, pz], dtype=float)
    k_norm = np.linalg.norm(k)
    if k_norm == 0:
        raise ValueError("Photon momentum cannot be zero")
    k = k / k_norm # Unit vector of photon momentum direction, used to construct transverse polarisation vectors orthogonal to it.

    # Arbitrary reference vector, z direction unless photon close to z-axis, then x to avoid 0 cross products.
    ref = np.array([0., 0., 1.]) if abs(k[2]) < 0.9 else np.array([1., 0., 0.])

    # Transverse basis (e1, e2) orthogonal to k.
    e1 = np.cross(k, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(k, e1)

    # Spin states
    if spin == 1:
        eps_spatial = (e1 + 1j * e2) / np.sqrt(2)
    elif spin == 0:
        eps_spatial = (e1 - 1j * e2) / np.sqrt(2)
    else:
        raise ValueError("Spin must be 0/down or 1/up")
    
    eps = np.array([0, *eps_spatial], dtype=complex).reshape(4,1) # reshapes from (4,) to (4,1) for matrix multiplication consistency

    # normalisation check: eps*_mu eps^mu = -|eps_spatial|^2 = -1  (Minkowski, all-spatial)
    eps_dot_eps = eps.conj().T @ metric @ eps
    if not np.isclose(eps_dot_eps, -1):
        raise ValueError(f"Photon polarization vector not properly normalised: eps*·eps = {eps_dot_eps}")

    # Enforce p·eps = 0 numerically (Ward identity).
    #p_dot_eps = p[0]*eps[0] - np.dot(p[1:], eps[1:])
    #eps = eps - (p_dot_eps / (p[0]**2 + np.dot(p[1:], p[1:]))) * p
    return eps

def ward_test(external_points, m_psi, e, photon_index):
    """
    Replace polarization vector with momentum and check amplitude → 0
    """
    test_points = []
    for i, pt in enumerate(external_points):
        new_pt = pt.copy()
        if i == photon_index:
            new_pt["force_momentum_polarization"] = True
        test_points.append(new_pt)
    return spin_averaged_matrix_element(test_points, m_psi, e)


def dirac_spinor_u(p, m_psi, spin):
    """
    Calculates the Dirac spinor u(p,s) for an external fermion current in the Dirac basis.

    Parameters:
    p (np.ndarray): four-momentum [E, px, py, pz], abs(E) input so crossing does not give complex spinors
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    spin (int): spin state (0/down or 1/up)

    Returns:
    u(p,s) (np.ndarray): external fermion current u in Dirac basis. Output shape (4,1).
    """
    px, py, pz = p[1:]
    E = np.sqrt(px**2 + py**2 + pz**2 + m_psi**2)
    chi = np.array([[1],[0]], dtype=complex) if spin == 1 else np.array([[0],[1]], dtype=complex)
    sigma_dot_p = np.array([[pz, px - 1j*py],
                            [px + 1j*py, -pz]], dtype=complex)
    
    eta = np.sqrt(E + m_psi)
    upper = eta * chi
    lower = (sigma_dot_p @ chi) / eta

    return np.vstack([upper, lower])

def dirac_spinor_v(p, m_psi, spin):
    """
    Calculates the Dirac spinor v(p,s) for an outgoing antifermion in the Dirac basis. Later conjugate and transpose to get vbar = v†γ^0 for the external antifermion current.

    Params:
    p (np.ndarray): four-momentum [E, px, py, pz], abs(E) input so crossing does not give complex spinors
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    spin (int): spin state (0/down or 1/up)

    Returns:
    v(p,s) (np.ndarray): outgoing antifermion current v in Dirac basis. Output shape (4,1)
    """
    px, py, pz = p[1:]
    E = np.sqrt(px**2 + py**2 + pz**2 + m_psi**2)
    chi = np.array([[1],[0]], dtype=complex) if spin == 1 else np.array([[0],[1]], dtype=complex)
    sigma_dot_p = np.array([[pz, px - 1j*py],
                            [px + 1j*py, -pz]], dtype=complex)

    eta = np.sqrt(E + m_psi)
    upper = (sigma_dot_p @ chi) / eta
    lower = eta * chi

    return np.vstack([upper, lower])

# FieldType class assigns names to indices of different fields.
class FieldType(Enum):
    A       = auto() # A_μ
    PSI     = auto() # ψ
    PSIBAR  = auto() # ψ̄ 

class Current:
    def __init__(self, field_type, current, momentum, flavour=None, fermion_id=None):
        """
        Created a current object for each external particle/combined current, and assign parameters as its values.

        Parameters:
        self: current object to be created
        field_type (FieldType): type of the current (A, PSI, PSIBAR)
        current (np.ndarray or float): the current value ( (4,) 4-vector for A and (4,1)/(1,4) spinors/matrices for PSI/PSIBARs)
        momentum (np.ndarray): four-momentum [E, px, py, pz]
        flavour (str): flavour of the current


        Returns:
        Current object with specified type, current value, and momentum.
        """
        self.type       = field_type
        self.current    = current
        self.p          = np.asarray(momentum, float)
        self.flavour    = flavour
        self.fermion_id = fermion_id

    def propagate(self, m_psi, epsilon = 1e-12):
        """
        Creates a new current object by applying the appropriate propagator to the new combined current, based on its field type.
        Premultiplies vector for ψ̄ and postmultiply for ψ, and postmultiply for matrix A_μ.

        Parameters:
        self (Current): current object to be propagated
        m_psi (float): mass of the fermion field ψ/ψ̄ 
        epsilon (float): imaginary part added to denominator to avoid limits in propagator, can made smaller if causing issues

        Returns:
        Modified current object with new current value, but unchanged type and momentum.
        """
        p_sq = self.p[0]**2 - np.dot(self.p[1:], self.p[1:])

        # Photon propagator: D(p) = (-i g^{μν}) / (p^2 + iε)
        if self.type == FieldType.A:
        #    metric = np.diag([1, -1, -1, -1])
            D_prop = (-1j * metric) / (p_sq + 1j * epsilon) # eps here is a small imiginary part, NOT the photon polarisation vector
            return Current(self.type, D_prop @ self.current, self.p, self.flavour, self.fermion_id)
        #    return Current(self.type, self.current / (p_sq + 1j * epsilon), self.p, self.flavour, self.fermion_id)

        # Dirac/vector/fermion propagator: S(p) = i (p-slash + m) / (p^2 - m^2 + iε)
        S_prop = 1j * (slash(self.p) + m_psi * np.eye(4, dtype=complex)) / (p_sq - m_psi**2 + 1j * epsilon)
        if self.type == FieldType.PSI:
            return Current(self.type, self.current @ S_prop, self.p, self.flavour, self.fermion_id)
        if self.type == FieldType.PSIBAR:
            return Current(self.type, S_prop @ self.current, self.p, self.flavour, self.fermion_id)

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

        # Enforce same fermion line + flavour
        def check_fermion(a, b):
            if a.fermion_id != b.fermion_id:
                raise ValueError("Different fermion lines")
            if a.flavour != b.flavour:
                raise ValueError("Different flavours")
        
        # ψ + ψ̄ → A_μ
        # J^mu =  ψ γ^mu ψ̄  (as ubar (ψ) and v (ψ̄ ) when outgoing)  with -ve vertex factor  -i e γ^mu
        if self.type == FieldType.PSI and other.type == FieldType.PSIBAR:
            check_fermion(self, other)
            J_mu = np.array([(self.current @ g @ other.current)[0,0]
                              for g in gammas], dtype=complex) # [0,0] extracts float from 1x1 array
            return Current(FieldType.A, -1j * e * J_mu, p_new, flavour=None, fermion_id=None) # A/photons have no flavour or fermion id
        if self.type == FieldType.PSIBAR and other.type == FieldType.PSI:
            check_fermion(self, other)
            J_mu = np.array([(other.current @ g @ self.current)[0,0]
                              for g in gammas], dtype=complex)
            return Current(FieldType.A, -1j * e * J_mu, p_new, flavour=None, fermion_id=None)

        # ψ + A_μ → ψ
        # Vertex: -i e γ^mu ε_mu  acting on the spinor
        # ε is contravariant and γ^mu ε_mu uses Minkowski metric implicitly:
        # γ^mu ε_mu = γ^0 ε^0 - γ^i ε^i   (lower with g_{munu} if done explicitly)
        def slash_eps(eps):
            eps = eps.flatten()
            # eps is contravariant 4-vector, returns return γ^mu ε_mu (lowered)
            return (gammas[0]*eps[0]
                    - gammas[1]*eps[1]
                    - gammas[2]*eps[2]
                    - gammas[3]*eps[3])
        if self.type == FieldType.PSI and other.type == FieldType.A:
            sl = slash_eps(other.current)
            # row: (1,4) @ (4,4)
            return Current(FieldType.PSI, -1j * e * (self.current @ sl), p_new, self.flavour, self.fermion_id) # keep flavour and fermion id as ψ
        if self.type == FieldType.A and other.type == FieldType.PSI:
            sl = slash_eps(self.current)
            return Current(FieldType.PSI, -1j * e * (other.current @ sl), p_new, other.flavour, other.fermion_id)

        # ψ̄ + A_μ → ψ̄
        if self.type == FieldType.PSIBAR and other.type == FieldType.A:
            sl = slash_eps(other.current)
            # column: (4,4) @ (4,1)
            return Current(FieldType.PSIBAR, -1j * e * (sl @ self.current), p_new, self.flavour, self.fermion_id) # keep flavour and fermion id as ψ̄ 
        if self.type == FieldType.A and other.type == FieldType.PSIBAR:
            sl = slash_eps(self.current)
            return Current(FieldType.PSIBAR, -1j * e * (sl @ other.current), p_new, other.flavour, other.fermion_id)


        raise ValueError(f"Invalid QED vertex: {self.type} + {other.type}")



# External currents, classified by field type and contains the spinor current (or 1 for scalars) and momentum information for the external particles.
def external_current(field_type, p, m_psi, incoming, spin=None, force_momentum_polarization=False, flavour=None, fermion_id=None):
    """
    Creates a current object for an external particle based on its field type, momentum, and spinor current (for fermions).

    Parameters:
    field_type (FieldType): type of the current (A, PSI, PSIBAR)
    p (np.ndarray): four-momentum [E, px, py, pz]
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    incoming (bool): whether the particle is incoming or outgoing, to determine which u/v spinor to construct
    spin (int): spin state (0/down or 1/up)

    Returns:
    Combined current object with new type, current value and momentum.
    """
    #print(f"field_type={field_type}, spin={spin}, incoming={incoming}")

    if field_type == FieldType.A:
        if force_momentum_polarization:
            cur = np.array(p, dtype=complex)  # Ward identity test
        else:
            cur = photon_polarization(p, spin) # A is the photon polarisation vector form incoming and outgoing photons shape (4,)
#    if field_type == FieldType.A:
#        eps = photon_polarization(p_spinor, spin) # photon polarisation vector
#        return Current(FieldType.A, eps, p)
        return Current(FieldType.A, cur, p, flavour, fermion_id)

    if field_type == FieldType.PSI:
        u = dirac_spinor_u(p, m_psi, spin) # u for incoming fermions shape (4,1)
        if not incoming:
            cur = (u.conj().T) @ gamma0 # ubar = u†γ^0 for outgoing fermions shape (1,4)
        else:
            cur = u
        return Current(FieldType.PSI, cur, p, flavour, fermion_id)

    if field_type == FieldType.PSIBAR:
        v = dirac_spinor_v(p, m_psi, spin) # v for outgoing antifermions shape (4,1)
        if incoming:
            cur = (v.conj().T) @ gamma0 # vbar = v†γ^0 for incoming antifermions shape (1,4)
        else:
            cur = v
        return Current(FieldType.PSIBAR, cur, p, flavour, fermion_id)

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
    Calculates the matrix element M for a given set of external particles with given spin state using B-G recursion.

    Parameters:
    external_currents (list of Current objects): list of external currents for each particle
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    e (float): coupling constant for QED interaction

    Returns:
    M (float): complex M for the given external particles and spin states
    """
    n = len(external_currents) - 1
    single_curr = external_currents[0]
    other_curr = external_currents[1:]
    J = {(i,): pt for i, pt in enumerate(other_curr)}

    # Single-particle currents
    for i, pt in enumerate(other_curr):
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
                try:
                    C = J[a].combine(J[b], e)

                    # Propagate only if not final off-shell current.
                    if no_pt < n and C.type != FieldType.A:
                        C = C.propagate(m_psi)

                    if total is None:
                        total = C
                    else:
                        if total.type != C.type:
                            raise ValueError("Type mismatch in current summation")
                        if total.fermion_id != C.fermion_id:
                            raise ValueError("Mixing fermion lines")
                        if total.flavour != C.flavour:
                            raise ValueError("Mixing flavours")
                        total = Current(C.type, total.current + C.current, C.p, total.flavour, total.fermion_id)

                except ValueError:
                    #print(f"fail2 for combination: {Ja.type} + {Jb.type}")
                    pass

            if total is not None:
                J[inds] = total
                print(total.type, inds, total.current)

    # Final off-shell current (all outgoing combined)
    final_curr = J[tuple(range(n))]


    # Final contraction: contracts n+1 particle off shell "final_curr" with "single_curr" to get M
    if (single_curr.type == FieldType.PSI and final_curr.type == FieldType.PSI
            and single_curr.flavour == final_curr.flavour
            and single_curr.fermion_id == final_curr.fermion_id): # single incoming u and final outgoing ubar
        M = (final_curr.current @ single_curr.current)[0,0]

    elif (single_curr.type == FieldType.PSIBAR and final_curr.type == FieldType.PSIBAR
            and single_curr.flavour == final_curr.flavour
            and single_curr.fermion_id == final_curr.fermion_id): # single incoming vbar and final outgoing v
        M = (single_curr.current @ final_curr.current)[0,0]

    elif single_curr.type == FieldType.A and final_curr.type == FieldType.A: #incoming photon and outgoing photon
        M = single_curr.current @ metric @ final_curr.current

    else:
        raise ValueError(f"Invalid final contraction structure {single_curr.type}, {single_curr.flavour}, {single_curr.fermion_id}, \nwith \n{final_curr.type}, {final_curr.flavour}, {final_curr.fermion_id}")
    print(f"M = {abs(M)}")
    return M

def spin_averaged_matrix_element(external_points, m_psi, e):
    """
    Calculates the squared and spin-averaged matrix element |M|^2 for a given set of external particles using B-G recursion.

    Constraints:
    Currently assumes only 2 spin states (0/up and 1/down).
    Currently assumes overall spin and fermion (baryon/lepton?) number conservation.

    Parameters:
    external_points (list of dict): list of external particles with their type and momentum, e.g. [{"type": FieldType.PSI, "p": p0}, ...]
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    e (float): coupling constant for QED interaction

    Returns:
    M_sq_av (float): squared and spin-averaged matrix element |M|^2 for the given external particles
    """
    # Determines the number of initial (anti-)fermions based on the external_points list, and generates all possible spin configurations for them.
    initial_indices = [i for i, pt in enumerate(external_points) if pt["incoming"]] # Now all incoming and outgoing particles have a spin as no scalars
    final_indices   = [i for i, pt in enumerate(external_points) if not pt["incoming"]]
    #print(len(initial_fermion_indices))
    all_indices     = initial_indices + final_indices
    spin_configs    = list(product([0,1], repeat=len(all_indices))) #Fermions and photons all have 2 spin states
    #print(spin_configs)

    crossed_points = []
    for i, pt in enumerate(external_points):
        new_pt      = pt.copy() # avoid changing original
        p           = new_pt["p"].copy()
        field_type  = new_pt["type"]
        incoming    = new_pt["incoming"]

        if incoming and i > 0: # crossing all but first incoming particle to have n+1 outgoing particles
            # Reverse 4-momentum.
            p = -p
            # Change field type (ψ ↔ ψ̄, A is unchanged)
            if field_type == FieldType.PSI:
                field_type = FieldType.PSIBAR
            elif field_type == FieldType.PSIBAR:
                field_type = FieldType.PSI
            crossed_points.append({"p": p, "type": field_type, "incoming": False,
                                   "flavour": pt.get("flavour"), "fermion_id": pt.get("fermion_id"),
                                   "force_momentum_polarization": pt.get("force_momentum_polarization", False)})
            continue
        crossed_points.append({"p": p, "type": field_type, "incoming": incoming,
                               "flavour": pt.get("flavour"), "fermion_id": pt.get("fermion_id"),
                               "force_momentum_polarization": pt.get("force_momentum_polarization", False)})
        
    # Loop over spins of (anti-)fermions
    total_M_sq = 0.0
    for spins in spin_configs:
        #print("Running spin config:", spin)
        spin_dict = dict(zip(all_indices, spins))

        external_currents = []

        for i, pt in enumerate(crossed_points):
            cur = external_current(pt["type"], pt["p"],
                                   m_psi, pt["incoming"], spin=spin_dict[i],
                                   flavour=pt.get("flavour"), fermion_id=pt.get("fermion_id"),
                                   force_momentum_polarization=pt.get("force_momentum_polarization", False))
            external_currents.append(cur) # From here incoming and spin is encoded in current, so no need to track.
        
        for xyz in external_currents:
            print(xyz.type, xyz.flavour, xyz.fermion_id, xyz.current.shape)
        print(spin_dict)

        M = calculate_amplitude(external_currents, m_psi, e)
        total_M_sq += abs(M)**2
    M_sq_av = total_M_sq / (2 ** len(initial_indices))
    return M_sq_av


if __name__ == "__main__":
    E = 1000
    m_psi = 0
    e = 1
    p = np.sqrt(E**2 - m_psi**2)

    p0 = np.array([E, 0, 0,  p])
    p1 = np.array([E, 0, 0, -p])
    eps = photon_polarization(p0, spin=0)
    eps1 = photon_polarization(p0, spin=1)

    if True:
        theta = np.pi / 4
        p2 = np.array([E,  p*np.sin(theta), 0,  p*np.cos(theta)])
        p3 = np.array([E, -p*np.sin(theta), 0, -p*np.cos(theta)])
        momenta22 = (p0, p1, p2, p3)
        external_points22 = [
            {"type": FieldType.PSI,     "p": p0, "incoming": True, "flavour": "e", "fermion_id": 0},
            {"type": FieldType.PSIBAR,  "p": p1, "incoming": True, "flavour": "e", "fermion_id": 0},
            {"type": FieldType.A,       "p": p2, "incoming": False},
            {"type": FieldType.A,       "p": p3, "incoming": False},
            ]
        
        print("Ward test photon 2:", ward_test(external_points22, m_psi, e, 2))
        print("Ward test photon 3:", ward_test(external_points22, m_psi, e, 3))
        #print(spin_averaged_matrix_element(external_points22, m_psi, e))
    
    if False:
        theta = np.pi / 4
        p2 = np.array([E,  p*np.sin(theta), 0,  p*np.cos(theta)])
        p3 = np.array([E, -p*np.sin(theta), 0, -p*np.cos(theta)])
        momenta22 = (p0, p1, p2, p3)
        external_points22 = [
            {"type": FieldType.PSI,     "p": p0, "incoming": True, "flavour": "e", "fermion_id": 0},
            {"type": FieldType.PSIBAR,  "p": p1, "incoming": True, "flavour": "e", "fermion_id": 0},
            {"type": FieldType.A,       "p": p2, "incoming": False, "flavour": "e", "fermion_id": 1},
            {"type": FieldType.A,       "p": p3, "incoming": False, "flavour": "e", "fermion_id": 1},
            ]
        
        print("Ward test photon 2:", ward_test(external_points22, m_psi, e, 2))
        print("Ward test photon 3:", ward_test(external_points22, m_psi, e, 3))
        #print(spin_averaged_matrix_element(external_points22, m_psi, e))




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
        print(spin_averaged_matrix_element(external_points23, m_psi, e))

