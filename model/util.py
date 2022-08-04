import numpy as np

def greedy(f,ntotal,nsensors):
    ''' A function to maximize function f(x), where only nsensors are 1.
    Input: 
    - ntotal: total number of measurements
    - nsensors: selected sensors
    Output: 
    A binary vector to represent the 

    Example Usage:
    - Simple linear case

    a = np.arange(10)
    f = lambda x: x.dot(a)
    x_sol = greedy(f,10,5)
    print(x_sol)   # It should select the last five sensors

    - Quadratic case

    A = np.array([[10,5,20,0], 
                  [5,10,0,0], 
                  [20,0,2,1],
                  [0,0,1,20]
                 ])
    f = lambda x: x.dot(A).dot(x)
    x_sol = greedy(f,4,2)
    print(x_sol) # Should be [1,0,0,1]
    f(x_sol)  # Should be 30, however, this is not the best we can do since greedy fails to find global optimum. 
    x_best = np.array([1,0,1,0])
    f(x_best) # Should be 52, which is larger than 30
    ''' 
    MIN_VALUES = np.NINF  # Number negative infinity
    x_sol = np.zeros(ntotal)
    for i_iter in range(nsensors):   # Search sensor one-by-one
        f_proposed = np.ones(ntotal)* MIN_VALUES
        for j in np.where(x_sol == 0)[0]:  # Search to move a non-zero value to one
            x_proposed = x_sol.copy()
            x_proposed[j] = 1
            f_proposed[j] = f(x_proposed)
        i_next = np.argmax(f_proposed)
        x_sol[i_next] = 1
        
    return x_sol

