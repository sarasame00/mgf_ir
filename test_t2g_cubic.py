import numpy as np 
import sparse_ir as ir  
from mgf_ir import hamiltonian, dyson, lattice
import parameters

'''
# Parameters loading section
fl = open("parameters_test_t2g_cubic")  # Opening the parameter file
lines = fl.readlines()  # Reading all lines from the file
fl.close()  # Closing the file

# Parse the parameter file to load the parameters
for line in lines:
    if line[:2] == "--":  # Check for lines starting with "--" (for float parameters)
        splitted = line[2:].split(' ')  # Split the line by spaces
        while True:
            try:
                splitted.remove('')  # Remove any empty strings from the list
            except:
                break
        locals()[splitted[0]] = float(splitted[2])  # Assign the value from the parameter to the variable name
        print(splitted[0] + "=" + splitted[2])  # Print the parameter name and its value
    
    if line[:2] == "++":  # Check for lines starting with "++" (for integer parameters)
        splitted = line[2:].split(' ')  # Split the line by spaces
        while True:
            try:
                splitted.remove('')  # Remove empty strings
            except:
                break
        locals()[splitted[0]] = int(splitted[2])  # Assign the integer value to the variable name
        print(splitted[0] + "=" + splitted[2])  

''' 

# Access each parameter and assign it in the current scope
for name in parameters.get_parameters_names():
    globals()[name] = getattr(parameters, name)
    print(f"{name}: {globals()[name]}")

print("Parameters set") 

def angular_momentum_t2g():
    '''
    Function to generate the angular momentum operators for the t$_{2g}$ orbitals
    '''
    a = np.arange(3)  # Create an array [0, 1, 2] to represent the x, y, z components
    y, x, z = np.meshgrid(a, a, a)  # Create a 3D meshgrid for these components
    
    # Forward (fwd) and backward (bkd) components based on the orbital symmetries
    fwd = ((x==0) * (y==1) * (z==2) + 
           (x==1) * (y==2) * (z==0) +
           (x==2) * (y==0) * (z==1))
    bkd = ((x==0) * (y==2) * (z==1) + 
           (x==1) * (y==0) * (z==2) +
           (x==2) * (y==1) * (z==0))
    
    return 1j * fwd - 1j * bkd  # Return the angular momentum matrix with an imaginary factor

# Assign the angular momentum components to lx, ly, lz
lx, ly, lz = angular_momentum_t2g()

# SPIN-ORBIT COUPLING
sx, sy, sz = np.array([[0.,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.eye(2)  # Defining spin matrices for x, y, z directions

# Compute the spin-orbit coupling Hamiltonian (HSOC) using Kronecker products
HSOC = 0.5 * xiSO * (np.kron(lx, sx) + np.kron(ly, sy) + np.kron(lz, sz))

# JEAN-TELLER EFFECT
# Computing the Jahn-Teller effect Hamiltonian (HJT) with Kronecker products for spin-1/2 particles
HJT = np.kron(-gJT / np.sqrt(3) * (lx @ lx - ly @ ly) * np.sin(theta) - gJT * (lz @ lz - 2/3*np.eye(3)) * np.cos(theta), np.eye(2))

# The local Hamiltonian is the sum of the spin-orbit coupling and Jahn-Teller effect
Hloc = HSOC + HJT

# INTERACTION WITH ELECTROMAGNETIC FIELDS (Moment tensor)
def moment_tensor(C, D):
    '''
    This function defines the hopping terms (tensor) between orbital components in a 3D lattice
    '''
    k = np.sqrt(3)  # A scaling factor
    t = np.zeros((3, 3, 3, 3), dtype=np.complex128)  # Initialize a 4D tensor with complex numbers to store the interactions

    # Define the hopping terms between the x, y, z components of the orbital angular momentum (indexed as 0, 1, 2)
    # with hopping directions (x, y, z) and p-orbitals (also indexed 0, 1, 2)
    
    # x component x hop direction
    t[0,0,1,2] = 0.5*C
    t[0,0,2,1] = 0.5*C

    # x component y hop direction
    t[0,1,1,2] = 0.5*D
    t[0,1,2,1] = 0.5*C

    # x component z hop direction
    t[0,2,1,2] = 0.5*C
    t[0,2,2,1] = 0.5*D

    # y component x hop direction
    t[1,0,0,2] = 0.5*D
    t[1,0,2,0] = 0.5*C
    
    # y component y hop direction
    t[1,1,0,2] = 0.5*C
    t[1,1,2,0] = 0.5*C

    # y component z hop direction
    t[1,2,0,2] = 0.5*C
    t[1,2,2,0] = 0.5*D

    # z component x hop direction
    t[2,0,0,1] = 0.5*D
    t[2,0,1,0] = 0.5*C

    # z component y hop direction
    t[2,1,0,1] = 0.5*C
    t[2,1,1,0] = 0.5*D

    # z component z hop direction
    t[2,2,0,1] = 0.5*C
    t[2,2,1,0] = 0.5*C

    return -1j*t  # Return the complex tensor representing the hopping terms

# Calculate the moment tensor with given constants C and D
P = moment_tensor(0.13, 0.05)

# Vector potential (describing an external field)
vec_pot = np.array([1, e0y*np.exp(1j*np.pi/180*phiy), e0z*np.exp(1j*np.pi/180*phiz)]) / np.sqrt(1+e0y**2+e0z**2)

# Calculate the virtual hopping interaction term using Einstein summation
virtual_hop = np.einsum('l,lkdp->kdp', vec_pot, P)

# Compute the hopping Hamiltonian (Hhop) using Kronecker products for spin-1/2 particles
Hhop = hoppampl * np.kron(np.einsum('kap,kbp->kab', virtual_hop, virtual_hop.conjugate()), np.eye(2))

# COULOMB INTERACTIONS

def coulomb_interaction_d_Oh(A, B, C):
    # This function computes the Coulomb interaction terms between the orbitals in the system
    
    F_4 = C/35
    F_2 = B + 5*F_4
    F_0 = A + 49*F_4
    
    F0 = F_0
    F2 = F_2*49
    F4 = F_4*441
    
    # Orbital basis transformation matrix
    sq2 = np.sqrt(2)
    orb_basis = np.array([[0, 0, -1j/sq2, 0, 1/sq2],
                          [1j/sq2, -1/sq2, 0, 0, 0],
                          [0, 0, 0, 1, 0],
                          [1j/sq2, 1/sq2, 0, 0, 0],
                          [0, 0, 1j/sq2, 0, 1/sq2]], dtype=np.complex128)
    
    # Slater integrals for Coulomb interaction
    c2 = np.array([[-2, np.sqrt(6), -2, 0, 0],
                    [-np.sqrt(6), 1, 1, -np.sqrt(6), 0],
                    [-2, -1, 2, -1, -2],
                    [0, -np.sqrt(6), 1, 1, -np.sqrt(6)],
                    [0, 0, -2, np.sqrt(6), -2]]) / 7

    c4 = np.array([[1, -np.sqrt(5), np.sqrt(15), -np.sqrt(35), np.sqrt(70)],
                    [np.sqrt(5), -4, np.sqrt(30), -np.sqrt(40), np.sqrt(35)],
                    [np.sqrt(15), -np.sqrt(30), 6, -np.sqrt(30), np.sqrt(15)],
                    [np.sqrt(35), -np.sqrt(40), np.sqrt(30), -4, np.sqrt(5)],
                    [np.sqrt(70), -np.sqrt(35), np.sqrt(15), -np.sqrt(5), 1]]) / 21

    # Initialize the Coulomb interaction matrix
    Vc = np.zeros((6, 6, 6, 6), dtype=np.complex128)
    
    # Loop through the Coulomb integrals and compute the interaction terms
    for i in range(6):
        oi = i//2
        si = i%2
        for j in range(6):
            oj = j//2
            sj = j%2
            for ok in range(3):
                sk = si # Spin delta on Coulomb integral
                k = 2*ok + sk
                for ol in range(3):
                    sl = sj # Spin delta on Coulomb integral
                    l = 2*ol + sl

                    if i==j or k==l:  # Pauli exclusion principle (no self-interactions)
                        continue

                    else: # Compute the Coulomb interaction terms
                        vi = orb_basis[:,oi]
                        vj = orb_basis[:,oj]
                        vk = orb_basis[:,ok]
                        vl = orb_basis[:,ol]

                        for m1 in range(-2,3):
                            for m2 in range(-2,3):
                                for m3 in range(-2,3):
                                    m4 = m1 + m2 - m3
                                    if m4>=-2 and m4<3:
                                        Vc[i,j,k,l] += vi[2-m1].conjugate()*vj[2-m2].conjugate()*vk[2-m3]*vl[2-m4] * (-1)**(m1+m3) * ( (m1==m3)*(m2==m4)*F0 + c2[2-m1,2-m3]*c2[2-m2,2-m4]*F2 + c4[2-m1,2-m3]*c4[2-m2,2-m4]*F4 )

                                        assert not np.any(np.isnan(Vc))  # Ensure no NaNs in the Coulomb interaction matrix
    return Vc.real  # Return the real part of the Coulomb interaction matrix


# Call the Coulomb interaction function with some parameters
V = coulomb_interaction_d_Oh(racahA, racahB, racahC)

H = hamiltonian.Hamiltonian(Hloc,V, Hhop)
irb = ir.FiniteTempBasis('F', beta, wM)
dy_solver = dyson.DysonSolver(H, irb, 4, lattice.Lattice(*(nk_lin,)*3))
dy_solver.solve()
A = dy_solver.spectral_function_mf()

'''
H = negf_ir.Hamiltonian(Hloc, V, Hhop)

irb = ir.FiniteTempBasis('F', beta, wM)
dy_solver = negf_ir.DysonSolver(H, irb, 4, negf_ir.Lattice(*(nk_lin,)*3))
dy_solver.solve()
A = dy_solver.spectral_function_mf()
'''