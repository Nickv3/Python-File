"""
Each observable has the same input of the generated 2D phase space array, and outputs a single value 

Done:
- cross_section
- scattering_angle (check and change)
- max_particle_energy
- invariant_mass (verify equation)

To Do:
- azimuthal_angle
"""


# For the scattering cross-section O(phi) = 1
def cross_section(p_list):
    return 1

# For the angle p1 is from the z-axis O(phi) = theta_p1
def scattering_angle(p_list):
    p_1v = p_list[1,1:]
    e_z = np.array([0,0,1])
    print(np.linalg.norm(p_1v))
    theta_p1 = np.arccos(np.dot(p_1v, e_z) / np.linalg.norm(p_1v))
    return theta_p1


# For the angle p1 is from the z-axis O(phi) = phi_p1
def azimuthal_angle(p_list):
    p_1v = p_list[1,1:]
    p_1v_con = p_1v * np.array([1,1,0]) # p_1 constrained to the x-y plane, i.e. z = 0
    e_x = np.array([1,0,0])
    print(np.linalg.norm(p_1v_con))
    return np.arccos(np.dot(p_1v_con, e_x) / np.linalg.norm(p_1v_con))


def max_particle_energy(p_list):
    return np.max(p_list[:,0])


def total_invariant_mass(p_list):
    #If total number of particles is 2 or fewer then only 1 is incoming, otherwise 2 are incoming
    if len(p_list) > 2:
        i = 2
    else:
        i = 1

    E = p_list[:,0]
    p_v = p_list[:,1:]

    E_term = (sum(E[0:i]) - sum(E[i:]))**2
    print(E_term)
    p_term = np.linalg.norm(sum(p_v[0:i]) - sum(p_v[i:]))**2
    print(p_term)
    return np.sqrt(E_term - p_term)


class observables:
    pass

import numpy as np
#print(scattering_angle(np.array([[1,2,3,4],[5,6,7,8]])))
#print(azimuthal_angle(np.array([[1,2,3,4],[5,6,7,8]])))
#print(max_particle_energy(np.array([[10,2,3,4],[5,6,7,8],[10.1,0,2,3]])))
#print(total_invariant_mass(np.array([[100,1,2,3],[50,4,5,6],[100,1,5,7]])))