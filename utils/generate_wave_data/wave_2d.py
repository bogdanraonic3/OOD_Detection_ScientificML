import numpy as np

# Generate Sine basis B of the shape (K, K, s, s)
# For each mode (i,j), B[i,j] represents the function 
# sin(np.pi*i*x)*np.sin(np.pi*j*y) evaluated at the s*s uniform grid
def generate_sine_basis(K = 16, s = 128):
    xx,yy = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    xx = xx/s
    yy = yy/s
    sine_basis  = np.zeros((K,K,s,s))
    for i in range(1,K+1):
        for j in range(1,K+1):  
            sine_basis[i-1,j-1] = np.sin(np.pi*i*xx)*np.sin(np.pi*j*yy)
    return sine_basis

# Generate matrix B of the shape (K, K)
# Each element (i,j) of the matrih has the value (i+1)**2 + (j+1)**2
def generate_square_matrix(K = 16):
    M = np.zeros((K, K))
    for i in range(1, K+1):
        for j in range(1, K+1):
            M[i-1, j-1] = i**2 + j**2
    return M

# Analytical solution of 2d Wave equation
def generate_solution_wave(coeff, time, sine_basis, square_matrix, K = 16, decay = 0.8, c = 0.1):
    multiplier_f = coeff * np.power(square_matrix, decay)
    multiplier_f = multiplier_f.reshape(K, K , 1, 1)
    f = np.pi*np.sum(multiplier_f * sine_basis, axis = (0,1))

    square_matrix_time = np.cos(c * np.pi * time * np.sqrt(square_matrix))
    multiplier_u = coeff * np.power(square_matrix, decay) * square_matrix_time
    multiplier_u = multiplier_u.reshape(K, K , 1, 1)
    u = np.pi*np.sum(multiplier_u * sine_basis, axis = (0,1))

    return f, u

# Generation of micro-macro perturbations
def generate_perturbation_wave(coeff, time, sine_basis, square_matrix, K = 16, decay = 0.8, c =0.1, perturbation = 0.25):
    p_matrix = np.random.uniform(-perturbation, perturbation, (K, K))
    #print(p_matrix)
    return generate_solution_wave(coeff+p_matrix , time, sine_basis, square_matrix, K = K, decay = decay, c = c)

def generate_training_data(c = 0.1, time = 5.0, N_data = 1024, K = 24, decay = -0.8, s = 128):
    sine_basis = generate_sine_basis(K, s)
    square_matrix = generate_square_matrix(K)

    inputs = np.zeros((N_data, s, s))
    targets = np.zeros((N_data, s, s))
    
    for n in range(N_data):
        coeff = np.random.uniform(-1,1, (K, K))
        inp, out = generate_solution_wave(coeff, time, sine_basis, square_matrix, K = K, decay = decay, c = c)
        inputs[n] = inp
        targets[n] = out

        if n%20 == 0:
            print(f"Done {n+1} out of {N_data}")
        
    print(" ")

    return inputs, targets

#-------------------------
def zero_out_elements(A, K_filter):
    """
    Sets to zero all elements of a 4D numpy array A 
    where the first or second index is greater than K.

    Args:
        A: The input numpy array of shape (K1, K1, s, s).
        K_filter: The integer threshold for the first two indices.

    Returns:
        A numpy array with the specified elements set to zero.
    """
    K1, _ = A.shape
    mask = np.logical_or(np.arange(K1) > K_filter, np.arange(K1)[:, np.newaxis] > K_filter)
    A[mask] = 0
    return A

def generate_training_data_varying_param(c = 0.1, 
                                        time = 5.0, 
                                        N_data = 1024, 
                                        K_min = 16,
                                        K_max = 28,
                                        decay_min = -0.65, 
                                        decay_max = -0.95,
                                        s = 128):

    sine_basis = generate_sine_basis(K_max, s) #((K,K,s,s))
    square_matrix = generate_square_matrix(K_max) #(K, K)

    inputs = np.zeros((N_data, s, s))
    targets = np.zeros((N_data, s, s))

    decays = np.zeros((N_data,))
    Ks = np.zeros((N_data,))
    coefficients = np.zeros((N_data, K_max, K_max))

    for n in range(N_data):
        K_filter = np.random.randint(K_min, K_max+1)
        decay = np.random.uniform(decay_min, decay_max)
        coeff = np.random.uniform(-1,1, (K_max, K_max))
        coeff_filter = zero_out_elements(coeff, K_filter)

        inp, out = generate_solution_wave(coeff_filter, time, sine_basis, square_matrix, K = K_max, decay = decay, c = c)
        inputs[n] = inp
        targets[n] = out

        coefficients[n] = coeff
        Ks[n] = K_filter
        decays[n] = decay

        if n%20 == 0:
            print(f"Done {n+1} out of {N_data}")
        
    print(" ")

    return inputs, targets, coefficients, Ks, decays

def generate_micro_macro_data(coeff_macro, perturbation = 0.25, c = 0.1, time = 5.0, N_data = 1024, K = 24, decay = -0.8, s = 128):
    sine_basis = generate_sine_basis(K, s)
    square_matrix = generate_square_matrix(K)

    inputs = np.zeros((N_data, s, s))
    targets = np.zeros((N_data, s, s))
    
    for n in range(N_data):
        inp, out = generate_perturbation_wave(coeff_macro, time, sine_basis, square_matrix, K = K, decay = decay, c = c, perturbation = perturbation)
        inputs[n] = inp
        targets[n] = out

        if n%20 == 0:
            print(f"Done {n+1} micro out of {N_data}")

    print(" ")

    return inputs, targets

