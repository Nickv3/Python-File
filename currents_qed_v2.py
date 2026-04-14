import numpy as np
from itertools import combinations, product
from enum import Enum, auto
from functools import lru_cache



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


# Set of constants created now to avoid repeated creation during recursion:
gamma0, gamma1, gamma2, gamma3 = gamma_matrices()
gammas = [gamma0, gamma1, gamma2, gamma3]   # gamma^mu
gamma_array = np.array(gammas)              # shape (4,4,4)
metric = np.diag([1., -1., -1., -1.])       # g^{mu nu}

ref_candidates = [
    np.array([1., 0., 0.]),
    np.array([0., 1., 0.]),
    np.array([0., 0., 1.])] # Used to construct transverse polarisation vectors for photons.

chi_candidates = {
    1: np.array([[1],[0]], dtype=complex),
    0: np.array([[0],[1]], dtype=complex)} # 1/up and 0/down spinors used in fermion construction.

sqrt2_inv = 1.0 / np.sqrt(2) # Used lots in normalisation.


# Early functions are simple arithmetic functions for optimisation, later ones are the core structure of the code for building currents.
def cross_product(a, b):
    """ Computes the cross product of two 3-vectors a and b."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]], dtype=float)


def slash_eps(eps):
    """Lowers contravariant 4-vector "eps", returns return γ^mu ε_mu."""
    return (gammas[0]*eps[0]
            - gammas[1]*eps[1]
            - gammas[2]*eps[2]
            - gammas[3]*eps[3])


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


def photon_polarization(p, spin):
    kx, ky, kz = p[1], p[2], p[3]
    k_norm = np.sqrt(kx*kx + ky*ky + kz*kz)
    if k_norm == 0:
        raise ValueError("Photon momentum cannot be zero")
    kx /= k_norm; ky /= k_norm; kz /= k_norm # Unit vector of photon momentum direction, used to construct transverse polarisation vectors orthogonal to it.
    k_arr = np.array([kx, ky, kz])

    # Chooses a stable reference vector that is not parallel to k to construct the transverse polarisation vectors.
    for ref in ref_candidates:
        c = cross_product(k_arr, ref)
        c_norm = np.sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2])
        if c_norm > 1e-6:
            break

    # Transverse basis (e1, e2) orthogonal to k constructed from reference vector.
    e1 = c / c_norm
    e2 = cross_product(k_arr, e1)

    # Check right-handedness of the basis:
    e1xe2 = cross_product(e1, e2)
    if e1xe2[0]*kx + e1xe2[1]*ky + e1xe2[2]*kz < 0:
        e2 = -e2

    # Spin states
    if spin == 1:
        eps_spatial = (e1 + 1j * e2) * sqrt2_inv
    elif spin == 0:
        eps_spatial = (e1 - 1j * e2) * sqrt2_inv
    else:
        raise ValueError("Spin must be 0/down or 1/up")
    
    return np.array([0, eps_spatial[0], eps_spatial[1], eps_spatial[2]], dtype=complex) # shape (4,)

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
    p (np.ndarray): four-momentum [E, px, py, pz], E always positive input so crossing does not give complex spinors
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    spin (int): spin state (0/down or 1/up)

    Returns:
    u(p,s) (np.ndarray): external fermion current u in Dirac basis. Output shape (4,1).
    """
    E, px, py, pz = p
    chi = chi_candidates[spin]
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
    p (np.ndarray): four-momentum [E, px, py, pz], E always positive input so crossing does not give complex spinors
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    spin (int): spin state (0/down or 1/up)

    Returns:
    v(p,s) (np.ndarray): outgoing antifermion current v in Dirac basis. Output shape (4,1)
    """
    E, px, py, pz = p
    chi = chi_candidates[spin]
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

# Constants used to constrain valid vertices and hence fermion flow:
allowed_vertices = frozenset({
    (FieldType.PSI,    FieldType.A),
    (FieldType.A,      FieldType.PSI),
    (FieldType.PSIBAR, FieldType.A),
    (FieldType.A,      FieldType.PSIBAR),
    (FieldType.PSI,    FieldType.PSIBAR),
    (FieldType.PSIBAR, FieldType.PSI)})


class Current:
    __slots__ = ('type', 'current', 'p', 'flavour')

    def __init__(self, field_type, current, momentum, flavour=None):
        """
        Created a current object for each external particle/combined current, and assign parameters as its values.

        Parameters:
        self: current object to be created
        field_type (FieldType): type of the current (A, PSI, PSIBAR)
        current (np.ndarray or float): the current value ((4,) 4-vector for A and (4,1)/(1,4) spinors/matrices for PSI/PSIBARs)
        momentum (np.ndarray): four-momentum [E, px, py, pz]
        flavour (str): flavour of the current

        Returns:
        Current object with specified type, current value, momentum and flavour.
        """
        self.type       = field_type
        self.current    = current   # (4,) A or [(4,1) or (1,4)]ψ/ψ̄  array depending on field type
        self.p          = np.asarray(momentum, float)
        self.flavour    = flavour

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
        p = self.p
        p_sq = p[0]*p[0] - p[1]*p[1] - p[2]*p[2] - p[3]*p[3]

        # Photon propagator: D(p) = (-i g^{μν}) / (p^2 + iε)
        if self.type == FieldType.A:
            D_prop = -1j / (p_sq + 1j * epsilon) # Scalar propagator as metric handled elsewhere.
            return Current(FieldType.A, D_prop * self.current, p, self.flavour)

        # Dirac/vector/fermion propagator: S(p) = i (p-slash + m) / (p^2 - m^2 + iε)
        S_prop = 1j * (slash(p) + m_psi * np.eye(4, dtype=complex)) / (p_sq - m_psi**2 + 1j * epsilon) # Shape (4,4) matrix
        if self.type == FieldType.PSI:
            return Current(FieldType.PSI, self.current @ S_prop, p, self.flavour)
        if self.type == FieldType.PSIBAR:
            return Current(FieldType.PSIBAR, S_prop @ self.current, p, self.flavour)

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
        Combined current object with new type, current value, momentum and flavour.
        """
        p_new = self.p + other.p
        st, ot = self.type, other.type
        
        # ψ + ψ̄ → A_μ
        # J^mu =  ψ γ^mu ψ̄  (as ubar (ψ) and v (ψ̄ ) when outgoing)  with -ve vertex factor  -i e γ^mu
        if st == FieldType.PSI and ot == FieldType.PSIBAR:
            if self.flavour != other.flavour:
                raise ValueError("Different flavours")
            J_mu = np.einsum('ij,mjk,kl->m', self.current, gamma_array, other.current)
            return Current(FieldType.A, -1j * e * J_mu, p_new, flavour=None) # A/photons have no flavour
        if st == FieldType.PSIBAR and ot == FieldType.PSI:
            if self.flavour != other.flavour:
                raise ValueError("Different flavours")
            J_mu = np.einsum('ij,mjk,kl->m', other.current, gamma_array, self.current)
            return Current(FieldType.A, -1j * e * J_mu, p_new, flavour=None)

        # ψ + A_μ → ψ
        # Vertex: -i e γ^mu ε_mu  acting on the spinor
        # ε is contravariant and γ^mu ε_mu uses Minkowski metric implicitly:
        # γ^mu ε_mu = γ^0 ε^0 - γ^i ε^i   (lower with g_{munu} if done explicitly)
        if st == FieldType.PSI and ot == FieldType.A:
            sl = slash_eps(other.current)
            return Current(FieldType.PSI, -1j * e * (self.current @ sl), p_new, self.flavour) # Row: (1,4) @ (4,4). Keep flavour and fermion id as ψ
        if st == FieldType.A and ot == FieldType.PSI:
            sl = slash_eps(self.current)
            return Current(FieldType.PSI, -1j * e * (other.current @ sl), p_new, other.flavour)

        # ψ̄ + A_μ → ψ̄
        if st == FieldType.PSIBAR and ot == FieldType.A:
            sl = slash_eps(other.current)
            return Current(FieldType.PSIBAR, -1j * e * (sl @ self.current), p_new, self.flavour) # Column: (4,4) @ (4,1). Keep flavour and fermion id as ψ̄ 
        if st == FieldType.A and ot == FieldType.PSIBAR:
            sl = slash_eps(self.current)
            return Current(FieldType.PSIBAR, -1j * e * (sl @ other.current), p_new, other.flavour)


        raise ValueError(f"Invalid QED vertex: {st} + {ot}")



# External currents, classified by field type and contains the spinor current (or 1 for scalars) and momentum information for the external particles.
def external_current(field_type, p, m_psi, incoming, spin=None, flavour=None, crossed=False, force_momentum_polarization=False):
    """
    Creates a current object for an external particle based on its field type, momentum, and spinor current (for fermions).

    Parameters:
    field_type (FieldType): type of the current (A, PSI, PSIBAR)
    p (np.ndarray): four-momentum [E, px, py, pz]
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    incoming (bool): whether the particle is incoming or outgoing, to determine which u/v spinor to construct
    spin (int): spin state (0/down or 1/up)
    flavour (str): flavour of the current (e/mu/tau for fermions, None for photons)
    crossed (bool): whether particle has had crossing applied
    force_momentum_polarization (bool): whether to use momentum for polarization (for Ward identity test)


    Returns:
    Combined current object with new type, current value and momentum.
    """
    #print(f"field_type={field_type}, spin={spin}, incoming={incoming}")

    if field_type == FieldType.A:
        if force_momentum_polarization:
            cur = np.array(p, dtype=complex) # Ward identity test
        else:
            cur = photon_polarization(-p, spin) if crossed else photon_polarization(p, spin) # A is the photon polarisation vector from incoming and outgoing photons shape (4,) NOTE: check p/-p
            if not incoming:
                cur = cur.conj()
        return Current(FieldType.A, cur, p, flavour)

    if field_type == FieldType.PSI:
        if crossed:
            # Use v spinor instead of u for a crossed antifermion
            v = dirac_spinor_v(-p, m_psi, spin)
            cur = (v.conj().T) @ gamma0
        else:
            u = dirac_spinor_u(p, m_psi, spin) # u for incoming fermions shape (4,1)
            cur = u if incoming else (u.conj().T) @ gamma0 # ubar = u†γ^0 for outgoing fermions shape (1,4)
        return Current(FieldType.PSI, cur, p, flavour)

    if field_type == FieldType.PSIBAR:
        if crossed:
            # Use u spinor instead of v for a crossed fermion
            cur = dirac_spinor_u(-p, m_psi, spin)
        else:
            v = dirac_spinor_v(p, m_psi, spin) # v for outgoing antifermions shape (4,1)
            cur = v if not incoming else (v.conj().T) @ gamma0 # vbar = v†γ^0 for incoming antifermions shape (1,4)
        return Current(FieldType.PSIBAR, cur, p, flavour)

    raise ValueError("Unknown field type")


@lru_cache(maxsize=None)
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

    #for c in external_currents:
    #    print(c.type, c.current, c.flavour, c.p)

    # Single Particle Currents:
    J = {(i,): pt for i, pt in enumerate(other_curr)}

    # Build multi-particle currents
    for no_pt in range(2, n + 1): # no_pt is the number of particles in the current being constructed, starting from 2 up to n
        for inds in combinations(range(n), no_pt): # indices of the external particles included in the current being constructed
            total = None
            for a, b in index_subsets(inds):
                if a not in J or b not in J:
                    #print("fail 1")
                    continue

                Ja, Jb = J[a], J[b]

                # Enforces fermion flow via testing for valid vertices.
                if (Ja.type, Jb.type) not in allowed_vertices:
                    continue

                # Enforces same flavour for ψ + ψ̄ 
                if Ja.type != FieldType.A and Jb.type != FieldType.A:
                    if Ja.flavour != Jb.flavour:
                        continue
                
                try:
                    C = Ja.combine(Jb, e)
                    #print(f"  Valid combination: {Ja.type}+{Jb.type}, a={a}, b={b}")

                    # Propagate all except final off-shell current.
                    if no_pt < n:
                        C = C.propagate(m_psi)

                    if total is None:
                        total = C
                    else:
                        if total.type != C.type:
                            raise ValueError("Type mismatch in summation")
                        if total.flavour != C.flavour:
                            raise ValueError("Mixing flavours in summation")
                        total = Current(C.type, total.current + C.current, C.p, C.flavour)

                except ValueError:
                    #print(f"fail2 for combination: {Ja.type} + {Jb.type}")
                    pass

            if total is not None:
                J[inds] = total
                #print(total.type, inds, total.current)

    # Final off-shell current (all outgoing combined)
    final_curr = J[tuple(range(n))]

    # Final contraction: contracts n+1 particle off shell "final_curr" with "single_curr" to get M
    if (single_curr.type == FieldType.PSI and final_curr.type == FieldType.PSI
            and single_curr.flavour == final_curr.flavour): # single incoming u and final outgoing ubar
        M = (final_curr.current @ single_curr.current)[0,0]

    elif (single_curr.type == FieldType.PSIBAR and final_curr.type == FieldType.PSIBAR
            and single_curr.flavour == final_curr.flavour): # single incoming vbar and final outgoing v
        M = (single_curr.current @ final_curr.current)[0,0]

    elif single_curr.type == FieldType.A and final_curr.type == FieldType.A: #incoming photon and outgoing photon
        M = single_curr.current @ metric @ final_curr.current

    else:
        raise ValueError(f"Invalid final contraction structure {single_curr.type}, {single_curr.flavour}, \nwith \n{final_curr.type}, {final_curr.flavour}")
    #print(f"M = {abs(M)}")
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
    crossed_points = []
    for i, pt in enumerate(external_points):
        p           = pt["p"].copy() # Leave original unchanged.
        field_type  = pt["type"]
        incoming    = pt["incoming"]

        if incoming and i > 0: # crossing all but first incoming particle to have n+1 outgoing particles
            # Reverse 4-momentum.
            p = -p
            # Change field type (ψ ↔ ψ̄, A is unchanged)
            if field_type == FieldType.PSI:
                field_type = FieldType.PSIBAR
            elif field_type == FieldType.PSIBAR:
                field_type = FieldType.PSI
            crossed_points.append({"p": p, "type": field_type, "incoming": False,
                                   "flavour": pt.get("flavour"),
                                   "crossed": True, # Used to construct correct spinor type for crossed fermions
                                   "force_momentum_polarization": pt.get("force_momentum_polarization", False)})
            continue
        crossed_points.append({"p": p, "type": field_type, "incoming": incoming,
                               "flavour": pt.get("flavour"),
                               "crossed": False,
                               "force_momentum_polarization": pt.get("force_momentum_polarization", False)})

    # Pre-extract fields to avoid repeated dict lookups in hot loop
    pt_types    = [pt["type"] for pt in crossed_points]
    pt_momenta  = [pt["p"] for pt in crossed_points]
    pt_incoming = [pt["incoming"] for pt in crossed_points]
    pt_flavour  = [pt["flavour"] for pt in crossed_points]
    pt_crossed  = [pt["crossed"] for pt in crossed_points]
    pt_force    = [pt["force_momentum_polarization"] for pt in crossed_points]
    n_particles = len(crossed_points)

    # Determines the number of initial (anti-)fermions based on the external_points list, and generates all possible spin configurations for them.
    all_indices     = [i for i in range(n_particles)]
    n_spins         = len(all_indices)
    initial_indices = sum(1 for i, pt in enumerate(external_points) if pt["incoming"]) # All incoming and outgoing particles have a spin as no scalars
    spin_configs    = list(product([0, 1], repeat=n_spins)) # Fermions and photons all have 2 spin states
    #print(spin_configs)

    precomputed = []
    for i in range(n_particles):
        spin_map = {}
        for s in (0, 1):
            spin_map[s] = external_current(pt_types[i], pt_momenta[i], m_psi,
                                           pt_incoming[i], spin=s,
                                           flavour=pt_flavour[i],
                                           crossed=pt_crossed[i],
                                           force_momentum_polarization=pt_force[i])
        precomputed.append(spin_map)

    # Loop over spins of (anti-)fermions
    total_M_sq = 0.0
    for spins in spin_configs:
        external_currents = [precomputed[i][spins[all_indices[i]]] for i in range(n_particles)]
        M = calculate_amplitude(external_currents, m_psi, e)
        total_M_sq += abs(M)**2
    #for spins in spin_configs:
    if False:
        #print("Running spin config:", spins)
        external_currents = []
        for i in range(n_particles):
            cur = external_current(pt_types[i], pt_momenta[i], m_psi,
                                   pt_incoming[i], spin=spins[all_indices[i]],
                                   flavour=pt_flavour[i],
                                   crossed=pt_crossed[i],
                                   force_momentum_polarization=pt_force[i])
            external_currents.append(cur) # From here incoming and spin is encoded in current, so no need to track.

        #for xyz in external_currents:
        #    print(xyz.type, xyz.flavour, xyz.current.shape)

        M = calculate_amplitude(external_currents, m_psi, e)
        total_M_sq += abs(M)**2
    M_sq_av = total_M_sq / (2 ** initial_indices)
    return M_sq_av

def matrix_element_ee_to_ee(p0, p1, k1, k2, m, e):
    """
    Spin-summed |M|^2 for e⁻(p0) e⁺(p1) → e⁻(k1) e⁺(k2) in QED.
    t-channel + s-channel photon exchange (Bhabha scattering).
    Averages over 4 initial spin states.
    """
    def photon_prop(q):
        q_sq = q[0]**2 - np.dot(q[1:], q[1:])
        return -1j / q_sq  # Feynman gauge: -i g^{mu nu} / q^2

    total = 0.0
    for s1, s2, s3, s4 in product([0,1], repeat=4):
        u_p0    = dirac_spinor_u(p0, m, s1)
        v_p1    = dirac_spinor_v(p1, m, s2)
        vbar_p1 = v_p1.conj().T @ gamma0
        u_k1    = dirac_spinor_u(k1, m, s3)
        ubar_k1 = u_k1.conj().T @ gamma0
        v_k2    = dirac_spinor_v(k2, m, s4)

        # t-channel (q = p0 - k1): photon exchanged between e- line and e+ line
        # [ū(k1)(-ieγ^μ)u(p0)] × (-ig_{μν}/q²) × [v̄(p1)(-ieγ^ν)v(k2)]
        # Contract over mu with metric: sum_mu η_μμ × (current1^μ)(current2^μ)
        q_t = p0 - k1
        D_t = photon_prop(q_t)
        j1_t = np.array([(ubar_k1 @ gammas[mu] @ u_p0)[0,0] for mu in range(4)])
        j2_t = np.array([(vbar_p1 @ gammas[mu] @ v_k2)[0,0] for mu in range(4)])
        M_t = (-1j*e)**2 * D_t * (j1_t @ metric @ j2_t)

        # s-channel (q = p0 + p1): e-e+ annihilate into photon, then photon → e-e+
        # [v̄(p1)(-ieγ^μ)u(p0)] × (-ig_{μν}/q²) × [ū(k1)(-ieγ^ν)v(k2)]
        q_s = p0 + p1
        D_s = photon_prop(q_s)
        j1_s = np.array([(vbar_p1 @ gammas[mu] @ u_p0)[0,0] for mu in range(4)])
        j2_s = np.array([(ubar_k1 @ gammas[mu] @ v_k2)[0,0] for mu in range(4)])
        M_s = (-1j*e)**2 * D_s * (j1_s @ metric @ j2_s)

        M = M_t - M_s  # relative minus from Fermi statistics
        total += abs(M)**2

    return total / 4.0


if __name__ == "__main__":
    E = 1000
    m_psi = 0
    e = 1
    p = np.sqrt(E**2 - m_psi**2)

    p0 = np.array([E, 0, 0,  p])
    p1 = np.array([E, 0, 0, -p])

    if True:
        theta = np.pi / 4
        p2 = np.array([E,  E*np.sin(theta), 0,  E*np.cos(theta)])
        p3 = np.array([E, -E*np.sin(theta), 0, -E*np.cos(theta)])
        external_points_ee_to_gg = [
            {"type": FieldType.PSI,     "p": p0, "incoming": True,  "flavour": "e"},
            {"type": FieldType.PSIBAR,  "p": p1, "incoming": True,  "flavour": "e"},
            {"type": FieldType.A,     "p": p2, "incoming": False},
            {"type": FieldType.A,  "p": p3, "incoming": False},
            ]

        #print("Ward test photon 2:", ward_test(external_points_ee_to_gg, m_psi, e, 2))
        #print("Ward test photon 3:", ward_test(external_points_ee_to_gg, m_psi, e, 3))        
        
        matrix_element_BG = spin_averaged_matrix_element(external_points_ee_to_gg, m_psi, e)
        
        #print(f"Matrix element from B-G recursion: {matrix_element_BG}")

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

if True:
    external_points_ee_to_ee = [
        {"type": FieldType.PSI,    "p": p0, "incoming": True,  "flavour": "e"},
        {"type": FieldType.PSIBAR, "p": p1, "incoming": True,  "flavour": "e"},
        {"type": FieldType.PSI,    "p": p2, "incoming": False,  "flavour": "e"},
        {"type": FieldType.PSIBAR, "p": p3, "incoming": False,  "flavour": "e"},
    ]

    bg      = spin_averaged_matrix_element(external_points_ee_to_ee, m_psi, e)
    feynman = matrix_element_ee_to_ee(p0, p1, p2, p3, m_psi, e)
    print(f"B-G:     {bg:.10f}")
    print(f"Feynman: {feynman:.10f}")