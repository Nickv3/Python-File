#Calculate the final matrix element for a set of external points
def calculate_matrix_element(external_points):
    combined_current  = calculate_combined_current(external_points[1:])
    matrix_element = 1 * 1 / calculate_propagator(external_points[1:]) * combined_current
    return abs(matrix_element)**2



#Calculate the combined current for a set of external points J_{12...n}
def calculate_combined_current(external_points, lambda_0 = 0.01):
    no_external_points = len(external_points)
    #print(no_external_points)
    J_values = {}
    #Create single particle currents
    J_1 = np.array([1] * no_external_points)
    J_values["J_1"] = J_1
    #print(J_values)

    #Create arbitrary dimensional particle currents
    for n in range(2, no_external_points + 1):
        J = np.zeros((no_external_points,) * n, dtype=complex)                                      #Initialize n-dimensional array for n-particle current
        for current_indices in combinations(range(no_external_points), n):                          #Iterate over all index combinations for n-particle current
            prop_external_points = tuple(external_points[i] for i in current_indices)               #Get the external points for the current index combination
            current_combinations = find_current_combinations(current_indices, n)                    #Find all possible current combinations that can form the n-particle current
            
            current_sum = 0
            for (pi_1, pi_2) in current_combinations:                                               #Iterate over all current combinations
                current_sum += J_values[f"J_{len(pi_1)}"][pi_1]*J_values[f"J_{len(pi_2)}"][pi_2]    #Sum the product of the two sub-currents
            #print(prop_external_points)
            #print(calculate_propagator(prop_external_points))
            J[current_indices] = calculate_propagator(prop_external_points) * (1j * lambda_0) * current_sum           #Calculate the n-particle current value
        J_values[f"J_{n}"] = J
    
    #print(J_values)
    return J_values[f"J_{no_external_points}"][current_indices]



def find_current_combinations(pi, n):
    seen = set()
    pi_pairs = []

    #Subset sizes only up to n//2 avoids duplicates
    for r in range(1, n // 2 + 1):
        for combo in combinations(pi, r):
            pi_1 = frozenset(combo)
            pi_2 = frozenset(pi) - pi_1

            pair = tuple(sorted([pi_1, pi_2], key=lambda s: (len(s), sorted(s)))) #Sort so A|B == B|A

            if pair not in seen:
                seen.add(pair)
                pi_pairs.append((tuple(sorted(pi_1)), tuple(sorted(pi_2))))
    return pi_pairs



#Calculate the propagator factor for a subset of the external points
def calculate_propagator(prop_external_points):
    E, px, py, pz = sum(prop_external_points)
    p_squared = E*E - (px*px + py*py + pz*pz)
    return 1j / (p_squared - m**2)
    


if True:
    #Import modules
    import numpy as np
    from itertools import combinations        #For generating index combinations for particle currents
    import matplotlib.pyplot as plt



    #Define phase space points
    E = 100
    p = 2
    m = p
    theta_list = np.linspace(0, np.pi, 10)
    
    y = []

    for theta in theta_list:
        p0 = np.array([E, 0, 0, p])
        p1 = -np.array([E, 0, 0, -p])
        p2 = np.array([E, p*np.sin(theta), 0, p*np.cos(theta)])
        p3 = np.array([E, -p*np.sin(theta), 0, -p*np.cos(theta)])
        #p4 = np.array([E, 0, p, 0])
        p_list = [p0, p1, p2, p3]
        print(p_list)



        #Run functions
        calculate_combined_current(p_list[1:])
        y.append(calculate_matrix_element(p_list))
    plt.plot(np.cos(theta_list), y)
    plt.xlabel("cos(theta)")
    plt.ylabel("|M|^2")
    plt.title("Matrix Element Squared vs cos(theta)")
    plt.figtext(0.5, 0, f"Energy = {E}, Momentum = {p}, Mass = {m}", ha="center", fontsize=10)
    plt.show()