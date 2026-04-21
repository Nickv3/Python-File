if True:
    #Import modules
    import numpy as np                      #General maths
    import matplotlib.pyplot as plt         #Plotting
    from currents_phi import calculate_matrix_element  #Matrix element calculation function



    #Define phase space points
    E = 100
    p = 4
    theta_list = np.linspace(0, np.pi, 100)
    theta_list = theta_list[3:-3]
    m = 0
    y = []

    for theta in theta_list:
        p0 = np.array([E, 0, 0, p])
        p1 = -np.array([E, 0, 0, -p])
        p2 = np.array([E, p*np.sin(theta), 0, p*np.cos(theta)])
        p3 = np.array([E, -p*np.sin(theta), 0, -p*np.cos(theta)])
        p4 = np.array([E, 0, p, 0])
        p_list = [p0, p1, p2, p3]
        #print(p_list)



        #Run functions
        y.append(calculate_matrix_element(p_list))
    if True:
        plt.plot(np.cos(theta_list), y)
        plt.xlabel("cos(theta)")
        plt.ylabel("|M|^2")
        plt.title("Matrix Element Squared vs cos(theta)")
        plt.figtext(0.5, 0, f"Energy = {E}, Momentum = {p}, Mass = {m}", ha="center", fontsize=10)
        plt.show()