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

chi_candidates = {
    1: np.array([[1],[0]], dtype=complex),
    0: np.array([[0],[1]], dtype=complex)} # 1/up and 0/down spinors used in fermion construction.

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
    p (np.ndarray): four-momentum [E, px, py, pz], E always positive input so crossing does not give complex spinors
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    spin (int): spin state (0 or 1)

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
    spin (int): spin state (0 or 1)

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

def merge_sign(labels_a, labels_b):
    """
    Get the sign from anticommuting fermion operators when placing A's operators before B's operators.
    Ensure the correct sign is applied based on the ordering of fermion labels as you would have when adding feynman diagrams

    Parameters:
    labels_a (tuple): fermion labels for current A
    labels_b (tuple): fermion labels for current B
    
    Returns:
    int: (-1)^(number of pairs (a,b) with a in labels_a, b in labels_b, a > b) 
    """
    count = sum(1 for a in labels_a for b in labels_b if a > b)
    return 1 if count % 2 == 0 else -1



# FieldType class assigns names to indices of different fields.
class FieldType(Enum):
    PHI = auto()    # φ
    PSI = auto()    # ψ
    PSIBAR = auto() # ψ̄ 

# Constants used to constrain valid vertices and hence fermion flow:
allowed_vertices = frozenset({
    (FieldType.PSI,    FieldType.PHI),
    (FieldType.PHI,      FieldType.PSI),
    (FieldType.PSIBAR, FieldType.PHI),
    (FieldType.PHI,      FieldType.PSIBAR),
    (FieldType.PSI,    FieldType.PSIBAR),
    (FieldType.PSIBAR, FieldType.PSI)})


class Current:
    __slots__ = ('type', 'current', 'p', 'flavour', 'fermion_labels')

    def __init__(self, field_type, current, momentum, flavour=None, fermion_labels=()):
        """
        Created a current object for each external particle/combined current, and assign parameters as its values.

        Parameters:
        self: current object to be created
        field_type (FieldType): type of the current (PHI, PSI, PSIBAR)
        current (np.ndarray or float): the current value (scalar for PHI and (4,1)/(1,4) spinors for PSI/PSIBARs)
        momentum (np.ndarray): four-momentum [E, px, py, pz]
        flavour (str): flavour of the current
        fermion_labels (tuple): tuple of fermion labels for the current, used to determine signs when combining currents with anticommuting fermions. Should be empty for photons.

        Returns:
        Current object with specified type, current value, momentum and flavour.
        """
        self.type = field_type
        self.current = current  # scalar φ or [(4,1) or (1,4)]ψ/ψ̄  array depending on field type
        self.p = np.asarray(momentum, float)
        self.flavour = flavour
        self.fermion_labels = fermion_labels

    def propagate(self, m_phi, m_psi, epsilon = 1e-12):
        """
        Creates a new current object by applying the appropriate propagator to the new combined current, based on its field type.
        Premultiplies line vector for ψ̄ and postmultiply for ψ, and scalar multiplication for φ.

        Parameters:
        self (Current): current object to be propagated
        m_phi (float): mass of the scalar field φ
        m_psi (float): mass of the fermion field ψ/ψ̄ 
        epsilon (float): imaginary part added to denominator to avoid limits in propagator, can made smaller if causing issues

        Returns:
        Modified current object with new current value, but unchanged type and momentum.
        """
        p = self.p
        p_sq = p[0]*p[0] - p[1]*p[1] - p[2]*p[2] - p[3]*p[3]

        # Scalar propagator: D(p) = i / (p^2 - m^2 + iε)
        if self.type == FieldType.PHI:
            D_prop = 1j / (p_sq - m_phi**2 + 1j * epsilon) #NOTE: check minus sign
            return Current(self.type, D_prop * self.current, p, self.flavour, self.fermion_labels)

        # Dirac/vector/fermion propagator: S(p) = i (p-slash + m) / (p^2 - m^2 + iε)
        S_prop = 1j * (slash(p) + m_psi * np.eye(4, dtype=complex)) / (p_sq - m_psi**2 + 1j * epsilon) # Shape (4,4) matrix
        if self.type == FieldType.PSI:
            return Current(FieldType.PSI, self.current @ S_prop, p, self.flavour, self.fermion_labels)
        if self.type == FieldType.PSIBAR:
            return Current(FieldType.PSIBAR, S_prop @ self.current, p, self.flavour, self.fermion_labels)

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
        Combined current object with new type, current value, momentum and flavour.
        """
        p_new = self.p + other.p
        st, ot = self.type, other.type

        # Compute fermion anticommutation sign and merged labels
        new_labels = tuple(sorted(self.fermion_labels + other.fermion_labels))
        sign = merge_sign(self.fermion_labels, other.fermion_labels)

        # ψ + ψ̄ → φ
        if st == FieldType.PSI and ot == FieldType.PSIBAR:
            if self.flavour != other.flavour:
                raise ValueError("Different flavours")
            vertex = 1j * g * (self.current @ other.current)[0,0] # [0,0] extracts float from 1x1 array
            return Current(FieldType.PHI, vertex, p_new, self.flavour, fermion_labels=new_labels)
        if st == FieldType.PSIBAR and ot == FieldType.PSI:
            if self.flavour != other.flavour:
                raise ValueError("Different flavours")
            vertex = 1j * g * (other.current @ self.current)[0,0]
            return Current(FieldType.PHI, vertex, p_new, self.flavour, fermion_labels=new_labels)

        # ψ + φ → ψ
        if st == FieldType.PSI and ot == FieldType.PHI:
            return Current(FieldType.PSI, 1j * g * self.current * other.current, p_new, self.flavour, fermion_labels=new_labels)
        if st == FieldType.PHI and ot == FieldType.PSI:
            return Current(FieldType.PSI, 1j * g * other.current * self.current, p_new, other.flavour, fermion_labels=new_labels)

        # ψ̄ + φ → ψ̄
        if st == FieldType.PSIBAR and ot == FieldType.PHI:
            return Current(FieldType.PSIBAR, 1j * g * self.current * other.current, p_new, self.flavour, fermion_labels=new_labels)
        if st == FieldType.PHI and ot == FieldType.PSIBAR:
            return Current(FieldType.PSIBAR, 1j * g * other.current * self.current, p_new, other.flavour, fermion_labels=new_labels)


        raise ValueError(f"Invalid Yukawa vertex: {st} + {ot}")



# External currents, classified by field type and contains the spinor current (or 1 for scalars) and momentum information for the external particles.
def external_current(field_type, p, m_psi, incoming, spin=None, flavour=None, crossed=False):
    """
    Creates a current object for an external particle based on its field type, momentum, and spinor current (for fermions).

    Parameters:
    field_type (FieldType): type of the current (PHI, PSI, PSIBAR)
    p (np.ndarray): four-momentum [E, px, py, pz]
    m_psi (float): mass of the fermion field ψ/ψ̄ 
    incoming (bool): whether the particle is incoming or outgoing, to determine which u/v spinor to construct
    spin (int): spin state (0 or 1)
    flavour (str): flavour of the current (e/mu/tau for fermions, None for PHI)
    crossed (bool): whether particle has had crossing applied

    Returns:
    Combined current object with new type, current value and momentum.
    """
    #print(f"field_type={field_type}, spin={spin}, incoming={incoming}")

    if field_type == FieldType.PHI:
        return Current(FieldType.PHI, 1.0 + 0j, p, flavour=None)

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



def calculate_amplitude(external_currents, m_phi, masses_psi, g):
    """
    Calculates the matrix element M for a given set of external particles with given spin state using B-G recursion.

    Parameters:
    external_currents (list of Current objects): list of external currents for each particle
    m_phi (float): mass of the scalar field φ
    masses_psi (dict): masses of the fermion field ψ/ψ̄  for different flavours
    g (float): coupling constant for Yukawa interaction

    Returns:
    M (float): complex M for the given external particles and spin states
    """
    n = len(external_currents) - 1
    single_curr = external_currents[0]
    other_curr = external_currents[1:]

    #for c in external_currents:
    #    print(c.type, c.current, c.flavour, c.p)

    # Assign fermion labels: unique integer per fermion, in order of appearance.
    # Bosons (photons) get empty labels.
    fermion_counter = 0
    if single_curr.type != FieldType.PHI:
        single_labels = (fermion_counter,)
        fermion_counter += 1
    else:
        single_labels = ()
    
    # Other Particle Currents (with fermion labels assigned):
    J = {}
    for i, pt in enumerate(other_curr):
        if pt.type != FieldType.PHI:
            fl = (fermion_counter,)
            fermion_counter += 1
        else:
            fl = ()
        J[(i,)] = Current(pt.type, pt.current, pt.p, pt.flavour, fermion_labels=fl)

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
                if Ja.type != FieldType.PHI and Jb.type != FieldType.PHI:
                    if Ja.flavour != Jb.flavour:
                        continue

                try:
                    C = Ja.combine(Jb, g)
                    #print(f"  Valid combination: {Ja.type}+{Jb.type}, a={a}, b={b}")

                    # Propagate all except final off-shell current.
                    if no_pt < n:
                        C = C.propagate(m_phi, masses_psi[C.flavour] if C.type != FieldType.PHI else 0)

                    if total is None:
                        total = C
                    else:
                        if total.type != C.type:
                            raise ValueError("Type mismatch in current summation")
                        if total.flavour != C.flavour:
                            raise ValueError("Mixing flavours in summation")
                        assert total.fermion_labels == C.fermion_labels, f"Fermion label mismatch: {total.fermion_labels} vs {C.fermion_labels}"
                        total = Current(C.type, total.current + C.current, C.p, C.flavour, C.fermion_labels)

                except ValueError:
                    #print(f"fail2 for combination: {Ja.type} + {Jb.type}")
                    pass

            if total is not None:
                J[inds] = total
                #print(total.type, inds, total.current)

    # Final off-shell current (all outgoing combined)
    final_curr = J[tuple(range(n))]

    # Final contraction: contracts n+1 particle off shell "final_curr" with "single_curr" to get M
    final_sign = merge_sign(final_curr.fermion_labels, single_labels)

    if (single_curr.type == FieldType.PSI and final_curr.type == FieldType.PSI 
        and single_curr.flavour == final_curr.flavour): # single incoming u and final outgoing ubar
        M = final_sign * (final_curr.current @ single_curr.current)[0,0]

    elif (single_curr.type == FieldType.PSIBAR and final_curr.type == FieldType.PSIBAR
          and single_curr.flavour == final_curr.flavour): # single incoming vbar and final outgoing v
        M = final_sign * (single_curr.current @ final_curr.current)[0,0]

    elif single_curr.type == FieldType.PHI and final_curr.type == FieldType.PHI: #incoming scalar and outgoing scalar
        M = final_sign * (single_curr.current * final_curr.current)

    else:
        raise ValueError(f"Invalid final contraction structure {single_curr.type}, {single_curr.flavour}, \nwith \n{final_curr.type}, {final_curr.flavour}")
    #print(f"M = {abs(M)}")
    return M

def spin_averaged_matrix_element(external_points, m_phi, masses_psi, g):
    """
    Calculates the squared and spin-averaged matrix element |M|^2 for a given set of external particles using B-G recursion.

    Constraints:
    Currently assumes only 2 spin states (0/up and 1/down).
    Currently assumes overall spin and fermion (baryon/lepton?) number conservation.

    Parameters:
    external_points (list of dict): list of external particles with their type and momentum, e.g. [{"type": FieldType.PSI, "p": p0}, ...]
    m_phi (float): mass of the scalar field φ
    masses_psi (dict): masses of the fermion field ψ/ψ̄  for different flavours
    g (float): coupling constant for Yukawa interaction

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
            # Change field type (ψ ↔ ψ̄, φ is unchanged)
            if field_type == FieldType.PSI:
                field_type = FieldType.PSIBAR
            elif field_type == FieldType.PSIBAR:
                field_type = FieldType.PSI
            crossed_points.append({"p": p, "type": field_type, "incoming": False, 
                                   "flavour": pt.get("flavour"), 
                                   "crossed": True}) # Used to construct correct spinor type for crossed fermions
            continue
        crossed_points.append({"p": p, "type": field_type, "incoming": incoming, 
                               "flavour": pt.get("flavour"),
                               "crossed": False})

    # Pre-extract fields to avoid repeated dict lookups in hot loop
    pt_types    = [pt["type"] for pt in crossed_points]
    pt_momenta  = [pt["p"] for pt in crossed_points]
    pt_incoming = [pt["incoming"] for pt in crossed_points]
    pt_flavour  = [pt["flavour"] for pt in crossed_points]
    pt_crossed  = [pt["crossed"] for pt in crossed_points]
    n_particles = len(crossed_points)

    # Determines the number of initial (anti-)fermions based on the external_points list, and generates all possible spin configurations for them.
    all_indices     = [i for i in range(n_particles)
                       if pt_types[i] != FieldType.PHI] # All fermions have spin, so include all indices of non-scalar particles
    n_spins         = len(all_indices)
    spin_position   = {idx: pos for pos, idx in enumerate(all_indices)}
    initial_indices = sum(1 for i, pt in enumerate(external_points)
                          if pt["type"] != FieldType.PHI and pt["incoming"])
    spin_configs    = list(product([0, 1], repeat=n_spins)) # Fermions have 2 spin states
    #print(spin_configs)

    precomputed = {}
    for i in range(n_particles):
        if pt_types[i] == FieldType.PHI:
            precomputed[i] = external_current(pt_types[i], pt_momenta[i], masses_psi[pt_flavour[i]] if pt_flavour[i] is not None else 0,
                                            pt_incoming[i], 
                                            flavour=pt_flavour[i],
                                            crossed=pt_crossed[i])
        else:
            precomputed[i] = {
                s: external_current(pt_types[i], pt_momenta[i], masses_psi[pt_flavour[i]] if pt_flavour[i] is not None else 0,
                                    pt_incoming[i], spin=s, 
                                    flavour=pt_flavour[i],
                                    crossed=pt_crossed[i])
                    for s in (0, 1)}
        
    # Loop over spins of (anti-)fermions
    total_M_sq = 0.0
    for spins in spin_configs:
        external_currents = []
        for i in range(n_particles):
            if pt_types[i] == FieldType.PHI:
                external_currents.append(precomputed[i])
            else:
                external_currents.append(precomputed[i][spins[spin_position[i]]])
        M = calculate_amplitude(external_currents, m_phi, masses_psi, g)
        total_M_sq += abs(M)**2
    #for spins in spin_configs:
    if False:
        #print("Running spin config:", spins)
        external_currents = []
        for i in range(n_particles):
            cur = external_current(pt_types[i], pt_momenta[i], masses_psi[pt_flavour[i]] if pt_flavour[i] is not None else 0,
                                   pt_incoming[i], spin=spins[all_indices[i]],
                                   flavour=pt_flavour[i],
                                   crossed=pt_crossed[i])
            external_currents.append(cur) # From here incoming and spin is encoded in current, so no need to track.

        #for xyz in external_currents:
        #    print(xyz.type, xyz.flavour, xyz.current.shape)

        M = calculate_amplitude(external_currents, m_phi, masses_psi, g)
        total_M_sq += abs(M)**2
    M_sq_av = total_M_sq / (2 ** initial_indices)
    return M_sq_av


if __name__ == "__main__":
    E = 1000
    m_phi = 10
    masses_psi = {"electron": 0.000511, "muon": 0.10566, "tau": 1.77686}
    g = 1
    p = np.sqrt(E**2 - masses_psi["electron"]**2)

    p0 = np.array([E, 0, 0,  p])
    p1 = np.array([E, 0, 0, -p])
    if True:
        theta = np.pi / 4
        p2 = np.array([E,  p*np.sin(theta), 0,  p*np.cos(theta)])
        p3 = np.array([E, -p*np.sin(theta), 0, -p*np.cos(theta)])
        momenta22 = (p0, p1, p2, p3)
        external_points22 = [
            {"type": FieldType.PSI,    "p": p0, "incoming": True,  "flavour": "electron"},
            {"type": FieldType.PSIBAR, "p": p1, "incoming": True,  "flavour": "electron"},
            {"type": FieldType.PHI,    "p": p2, "incoming": False},
            {"type": FieldType.PHI,    "p": p3, "incoming": False},
            ]
        print(spin_averaged_matrix_element(external_points22, m_phi, masses_psi, g))

    if True:
        E_phi = 2*E/3
        k = np.sqrt(E_phi**2 - masses_psi["electron"]**2)
        p2 = np.array([E_phi,  k,                 0, 0])
        p3 = np.array([E_phi, -k/2,  np.sqrt(3)*k/2, 0])
        p4 = np.array([E_phi, -k/2, -np.sqrt(3)*k/2, 0])
        momenta23 = (p0, p1, p2, p3, p4)
        external_points23 = [
            {"type": FieldType.PSI,    "p": p0, "incoming": True,  "flavour": "electron"},
            {"type": FieldType.PSIBAR, "p": p1, "incoming": True,  "flavour": "electron"},
            {"type": FieldType.PHI,    "p": p2, "incoming": False},
            {"type": FieldType.PHI,    "p": p3, "incoming": False},
            {"type": FieldType.PHI,    "p": p4, "incoming": False}
            ]
        print(spin_averaged_matrix_element(external_points23, m_phi, masses_psi, g))