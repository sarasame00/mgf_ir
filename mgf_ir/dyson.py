import numpy as np
import sparse_ir as ir
from .lattice import Lattice
from .hamiltonian import Hamiltonian
from .mats_func import non_int_g, zeroG, mats_copy, zeroBubble, pol_bubble_from_green, MatsubaraGreen, MatsubaraGreenLatt



class DysonSolver:
    def __init__(self, H, ir_basis, N, lattice=None):
        """
        Initializes the DysonSolver class, which solves the Dyson equation for a given Hamiltonian H.
        The solver works in the Matsubara formalism and supports lattice-based systems (optional).
        """
        # Ensure that H is an instance of the Hamiltonian class
        if not isinstance(H, Hamiltonian):
            raise TypeError("H must be a Hamiltonian type")
        
        # Store the Hamiltonian, particle density, and the basis
        self.__H = H
        self.__N = N
        self.__basis = ir_basis
         
        # Check if a lattice is provided; set the lattice flag accordingly
        if isinstance(lattice, Lattice):
            self.__lattice = lattice
            self.__in_lattice = True
        else:
            self.__lattice = None
            self.__in_lattice = False
        
        self.__adjust_mu(is_interacting=False) # Adjust the chemical potential for non-interacting system
            
        # Initialize the non-interacting Green's function (G0)
        self.__G0 = non_int_g(self.H, self.mu, self.basis, self.lattice) if self.__in_lattice else non_int_g(self.H, self.mu, self.basis) 

        # Initialize Green's function on the lattice (if in lattice) or copy the non-interacting Green's function
        self.__Gk = mats_copy(self.__G0) if self.__in_lattice else None

        # Initialize the local Green's function (Gloc), depending on whether lattice is used
        self.__Gloc = MatsubaraGreen(np.sum(self.__Gk.Fl, axis=1)/self.nk, ir_basis) if self.__in_lattice else mats_copy(self.__G0)
        
        # Initialize the self-energy term (SE) to be zero (non-interacting)
        self.__SE = zeroG(H, ir_basis)

         # Initialize the self-energy for the Hartree-Fock (HF) approximation
        self.__sehf = 0
        
        # Initialize time and frequency samplers for the Matsubara Green's function
        self.__sampl_time = ir.TauSampling(ir_basis)
        self.__sampl_freq = ir.MatsubaraSampling(ir_basis)
        
        # Flag to track if the solver has finished solving the Dyson equation
        self.__solved = False
    
    @property
    def H(self):
        return self.__H # Return the Hamiltonian
    
    @property
    def particle_density(self):
        return self.__N # Return the particle density
    
    @property
    def dim(self):
        return self.H.dim # Return the dimension of the Hamiltonian (number of quantum states)
    
    @property
    def basis(self):
        return self.__basis  # Return the IR basis
    
    @property
    def beta(self):
        return self.basis.beta # Return the beta value of the basis
    
    @property
    def particle(self):
        return self.basis.statistics # Return the statistics (Fermionic or Bosonic) of the basis
    
    @property
    def lattice(self):
        if self.__in_lattice:
            return self.__lattice # Return the lattice if it's being used
        else:
            raise AttributeError("This Dyson solver has no lattice")
    
    @property
    def nk(self):
        return self.lattice.nk  # Return the number of lattice points (nk)
    
    @property
    def mu(self):
        return self.__mu # Return the chemical potential
    
    @property
    def g(self):
        return self.__G0 # Return the non-interacting Green's function
    
    @property
    def Gloc(self):
        return self.__Gloc # Return the local Green's function
    
    @property
    def Gk(self):
        if self.__in_lattice:
            return self.__Gk # Return the Green's function on the lattice if it's being used
        else:
            raise AttributeError("This Dyson solver has no lattice")
    
    @property
    def segy(self):
        return self.__SE # Return the self-energy (SE)
    
    @property
    def segy_hf(self):
        return self.__sehf # Return the self-energy in the Hartree-Fock approximation
    
    def __adjust_mu(self, th=1e-8, delta_mu=0.5, is_interacting=True):
        """
        Adjusts the chemical potential for the system (either non-interacting or interacting) until the desired particle density is reached.
        
        This function adjusts the chemical potential (mu) to match the desired particle density by iterating and computing the Green's function.
        It is used in both the non-interacting and interacting cases.
        
        Parameters:
        th (float): The threshold for the difference in particle density, below which the iteration stops.
        delta_mu (float): The initial step size used to adjust the chemical potential during each iteration.
        is_interacting (bool): Flag to indicate if the system is interacting or non-interacting.
        """
                
        self.__mu = 0 # Initialize the chemical potential (mu) to zero
        last_sign = 2  # Initialize the sign of the particle density difference to 2 (no previous value)
        loops = 0  # Loop counter for the number of iterations

        print("Adjusting chemical potential")

        while True:
            if is_interacting:
                self.__G0 = non_int_g(self.H, self.mu, self.basis, self.lattice) if self.__in_lattice else non_int_g(self.H, self.mu, self.basis)
                # Update the Green's function using the current self-energy terms (self.__SE)
                self.__update_green()
                # Compute the particle density (N) from the Green's function
                N = self.__eval_particle_dens()
            else:
                g0 = non_int_g(self.H, self.mu, self.basis, self.lattice) if self.__in_lattice else non_int_g(self.H, self.mu, self.basis)
                if self.__in_lattice:
                    N = -np.sum(g0.Ftau[-1]) / self.nk
                else:
                    N = -np.sum(g0.Ftau[-1])

            # Calculate the difference (DN) between the desired particle density and the computed density
            DN = self.__N - N
            
            # Check if the particle density difference is small enough (convergence)
            if abs(DN) < th:
                break  # If converged, exit the loop

            # Determine the sign of the density difference (whether the density needs to increase or decrease)
            sign_N = 1 if DN > 0 else -1
            
            # If the sign of DN changes, reduce the step size (delta_mu) to avoid overshooting
            if not (last_sign == sign_N or last_sign == 2):
                delta_mu /= 2

            loops += 1  # Increment the loop counter

            print(f"Iteration {loops} computed particle density of {N:.8f} and mu={self.__mu:.8f}")

            # Adjust the chemical potential (mu) based on the sign of DN (either increasing or decreasing)
            self.__mu += sign_N * delta_mu
            last_sign = sign_N  # Update the sign of DN for the next iteration

        if not is_interacting:
            print(f"Non-interacting Green's function initialized with mu={self.mu:.8f}")
        else:
            print(f"Green's function optimized with mu={self.mu:.8f}")

    
    def __eval_particle_dens(self):
        """
        Evaluate the particle density using the Green's function (G).
        """
        return -np.sum(self.Gloc(self.beta, 'time')) # Sum over the Green's function in imaginary time
    
    def __update_self_energy(self, approx):
        """
        Update the self-energy based on the chosen approximation (e.g., Hartree-Fock, GW).
        """
        ps = {'F':-1, 'B':1}[self.particle] # Particle statistics (negative for fermions, positive for bosons)
        self.__sehf = ps * self.H.Umf @ self.Gloc.Ftau[-1] # Update the self-energy in the Hartree-Fock approximation
        
        if approx == "HF":
            pass # Hartree-Fock approximation is already handled
        
        elif approx == "GW":
            # GW approximation (for self-energy) using polarization bubbles, not implemented
            pik = pol_bubble_from_green(self.Gk)
            raise NotImplementedError
        
        else:
            raise NotImplementedError
    
    def __update_green(self):
        """
        Update the Green's function using the Dyson equation. This involves updating both the local Green's function and Green's function on the lattice.
        """
        # ps = {'F':-1, 'B':1}[self.particle]
        # sigma_hf = ps * np.einsum('ab,lb,l->a', self.H.Umf, self.Gloc[:], self.basis.u(self.beta))
        
        if self.__in_lattice:
            # Update Green's function on the lattice using the self-energy
            Gkiw = (self.g.Fiw**-1 - self.segy_hf[None,None,:] - self.segy.Fiw[:,None,:])**-1
            self.__Gk[:] = self.__sampl_freq.fit(Gkiw, axis=0).real # Update Green's function in momentum space
            self.__Gloc[:] = np.sum(self.Gk[:], axis=1) / self.nk # Update local Green's function (average over momentum space)
        
        else:
            Giw = (self.g.Fiw**-1 - self.segy.Fiw)**-1 # Update Green's function in the absence of lattice
            self.__Gloc[:] = self.__sampl_freq.fit(Giw, axis=0).real # Update local Green's function
    
    def __checkN(self, N):
        """
        Check the particle density (N) from the local Green's function (Gloc).
        """
        return -np.sum(self.Gloc.Ftau[-1]) # Return the particle density from the local Green's function (in time space)
        # return -np.sum(self.Gloc(self.beta, 'time'), axis=-1)
    
    
    def solve(self, approx = "HF", th=1e-6, diis_mem=5):
        """
        Solves the Dyson equation self-consistently for the Green's function. This method iterates to find the self-consistent solution.
        The self-energy is updated at each iteration, and the chemical potential is adjusted to match the desired particle density.
        """
        print("Starting self-consistent solution of Dyson equation\n")

        loops = 0 # Initialize loop counter

        diis_err = np.zeros((diis_mem, self.basis.size+1, self.dim)) # Initialize memory for DIIS (Direct Inversion in Iterative Subspace) errors
        diis_val = np.zeros((diis_mem, self.basis.size+1, self.dim)) # Initialize memory for DIIS vector values

        while True:
            # Store the last value of the local Green's function (Gloc) for convergence check
            last_Glocl = np.copy(self.Gloc.Fl)

            # Update the self-energy based on the chosen approximation (e.g., Hartree-Fock)
            self.__update_self_energy(approx)

            # Prepare the DIIS vectors for error correction
            diis_vec = np.zeros((self.basis.size+1,self.dim))
            diis_vec[0] = np.copy(self.segy_hf)  # First vector corresponds to the Hartree-Fock self-energy
            diis_vec[1:] = np.copy(self.segy.Fl) # Other vectors correspond to the self-energy terms for the system
            
            # Shift previous DIIS values and errors to make space for the new ones
            diis_val[:-1] = np.copy(diis_val[1:])
            diis_val[-1] = np.copy(diis_vec)
            diis_err[:-1] = np.copy(diis_err[1:])
            diis_err[-1] = np.copy(diis_val[-1] - diis_val[-2]) # Compute the error for the current iteration

            # If the memory size for DIIS is reached, solve for new coefficients using the previous DIIS errors
            if loops >= diis_mem:
                B = np.zeros((diis_mem,)*2) # Initialize the matrix for DIIS linear system
                for i in range(diis_mem):
                    for j in range(i, diis_mem):
                        # Compute the inner products of the errors for each pair of iterations
                        B[i,j] = np.sum(diis_err[i] * diis_err[j])
                        if i != j:
                            B[j,i] = np.copy(B[i,j]) # Symmetric matrix
                
                B /= np.mean(B)
                try:
                    Binv = np.linalg.inv(B)
                except:
                    Binv = np.linalg.inv(B + np.eye(diis_mem)*1e-8)
                c_prime = Binv @ np.ones((diis_mem,))
                c = c_prime / np.sum(c_prime) # Normalize the coefficients
                diis_vec = np.sum(c[:,None,None]*diis_val, axis=0) # Compute the new DIIS vector by weighting previous values
                diis_val[-1] = np.copy(diis_vec) 
                diis_err[-1] = np.copy(diis_val[-1] - diis_val[-2])  # Update the error
                self.__segy = np.copy(diis_vec[0])  # Update the self-energy (HF part)
                self.__SE[:] = np.copy(diis_vec[1:])  # Update the self-energy (interacting part)

            # Adjust the chemical potential (mu) to match the desired particle density
            self.__adjust_mu()
            
            # Check the convergence of the Green's function by computing the difference between the current and last values
            conv = np.sqrt(np.sum((self.Gloc.Fl - last_Glocl)**2))
            
            loops += 1 # Increment the iteration counter
            # If the difference is small enough, stop the iterations
            if conv < th:
                break

            print("Iteration %i completed with convergence %.8f" % (loops, conv))
            print('-------------------------------------------------------\n')

        print("Finished after %i iterations with convergence of %.8f" % (loops, conv))

        self.__solved = True # Mark the solver as successfully solved
    
    def spectral_function_mf(self):
        """
        Calculates the spectral function using the mean-field approximation. 
        The spectral function describes the density of states of the system.
        """
        def rho(w, eta):
            """
            Computes the spectral density (A) at a given frequency w with a small damping factor eta.
            """
            # Ensure that eta (damping factor) is a constant number (either int or float)
            if not isinstance(eta, (int, float)):
                raise ValueError("Dumping must be a constant number")
            
            # If the frequency is a single value (not an array)
            if isinstance(w, (int, float)):
                # Check if the frequency is within the allowed range
                if w < -self.basis.wmax or w > self.basis.wmax:
                    raise ValueError("Not able to compute outside max frequency %.3feV" % self.basis.wmax)
                 # If lattice is used, calculate the spectral function using the lattice Green's function (Hk) and self-energy
                if self.__in_lattice:
                    A = eta/self.nk/np.pi * np.sum(((w - self.H.Hk(self.lattice) - self.segy_hf[None,:] + self.mu)**2 + eta**2)**-1, axis=0)
                else:
                    A = eta/np.pi * ((w - self.H.w - self.segy_hf + self.mu)**2 + eta**2)**-1
            
            # If the frequency is an array of values, handle it as an iterable
            else:
                try:
                    w = np.array(w)
                except:
                    raise TypeError("Frequencies must be an iterable of numbers")
                
                 # Check if any frequencies are out of range
                if np.any(w < -self.basis.wmax) or np.any(w > self.basis.wmax):
                    raise ValueError("Not able to compute outside max frequency %.3feV" % self.basis.wmax)
                
                # If lattice is used, calculate the spectral function over multiple frequencies
                if self.__in_lattice:
                    A = eta/self.nk/np.pi * np.sum(((w[:,None,None] - self.H.Hk(self.lattice)[None,:,:] - self.segy_hf[None,None,:] + self.mu)**2 + eta**2)**-1, axis=1)
                else:
                    A = eta/np.pi * ((w[:,None] - self.H.w[None,:] - self.segy_hf[None,:] + self.mu)**2 + eta**2)**-1
            
            # Return the spectral function (A) using the eigenbasis of the Hamiltonian
            return np.einsum('ij,...j,jk->...ik', self.H.P, A, self.H.Ph)
        
        return rho

    def particle_number(self):
        """
        Compute the particle number (density) from the local Green's function (G).
        """
        N = self.Gloc.Ftau[-1] # Get the last value of the local Green's function (in time space)
        # Calculate the particle number using the eigenvectors of the Hamiltonian
        return np.einsum('ij,j,jk->ik', self.H.P, N, self.H.Ph)

    def to_eigenbasis(self, A):
        """
        Transform a matrix A to the eigenbasis of the Hamiltonian using the eigenvectors (P, Ph).
        This is useful for converting between the physical basis and the diagonalized basis of the system.
        """
        # Perform the transformation to the eigenbasis using Einstein summation
        return np.einsum('ij,...jk,ki->...i', self.H.P, A, self.H.Ph)
    
    # def solve(self, approx = "HF", th=1e-6, delta_mu = 0.1, diis_mem=5):
    #     print("Starting self-consistent solution of Dyson equation\n")
    #     last_sign = 2
    #     while True: #Loop for mu
    #         loops = 0
    #         diis_err = np.zeros((diis_mem, self.basis.size, self.dim))
    #         diis_val = np.zeros((diis_mem, self.basis.size, self.dim))
    #         while True: #sc loop
    #             Glocl_prev = np.copy(self.Gloc.Fl)
    #             self.__update_self_energy(approx)
    #             self.__update_green()
    #             # diis_val[:-1] = np.copy(diis_val[1:])
    #             # diis_val[-1] = np.copy(self.Gloc.Fl)
    #             # diis_err[:-1] = np.copy(diis_err[1:])
    #             # diis_err[-1] = np.copy(self.Gloc.Fl - Glocl_prev)
    #             # if loops >= diis_mem:
    #             #     B = np.zeros((diis_mem,)*2)
    #             #     for i in range(diis_mem):
    #             #         for j in range(i, diis_mem):
    #             #             B[i,j] = np.sum(diis_err[i] * diis_err[j])
    #             #             if i != j:
    #             #                 B[j,i] = np.copy(B[i,j])
    #             #     c_prime = np.linalg.inv(B) @ np.ones((diis_mem,))
    #             #     c = c_prime / np.sum(c_prime)
    #             #     self.__Gloc[:] = np.sum(c[:,None,None]*diis_val, axis=0)
    #             #     diis_val[-1] = np.copy(self.Gloc.Fl)
    #             #     diis_err[-1] = np.copy(self.Gloc.Fl - Glocl_prev)
    #             conv = np.sqrt(np.sum((Glocl_prev - self.Gloc.Fl)**2))
    #             loops += 1
    #             print("Iteration %i completed with convergence %.8f" % (loops, conv))
    #             if conv < th:
    #                 break
    #         DN = self.__N - self.__eval_particle_dens()
    #         print("Convergence aquired for mu=%.8f" % self.mu)
    #         print("Particle density computed: %.8f" % (self.__N - DN))
            
    #         if abs(DN) < th:
    #             break
    #         sign_N = 1 if DN > 0 else -1
    #         if not (last_sign == sign_N or last_sign == 2):
    #             delta_mu /= 2
    #         self.__mu += sign_N * delta_mu
    #         last_sign = sign_N
            
    #         print("New mu=%.8f" % self.__mu)
    #         print('-------------------------------------------------------\n')
            
    #         self.__G0 = non_int_g(self.H, self.mu, self.basis, self.lattice)
    #     print("Finished")
    #     self.__solved = True
    
    # def spectral_function(self, w):
    #     if not self.__solved:
    #         raise AttributeError("Not able to give the spectral function before solve Dyson equation")
        
    #     rhol = -np.einsum('ij,lj,jk->lik', self.H.P, self.Gloc.Fl/self.basis.s[:,None], self.H.Ph)
    #     poly = self.basis.v(w)
    #     return np.einsum('lij,l...->...ij', rhol, poly)
    
    # def spectral_function(self, size=201, quadrature_order=4):
    #     try:
    #         size = int(size)
    #     except:
    #         raise ValueError("Size must be an integer greater than 0")
        
    #     try:
    #         quadrature_order = int(quadrature_order)
    #     except:
    #         raise ValueError("Quadrature order must be an integer greater than 0")
        
    #     if size <= 0:
    #         raise ValueError("Size must be an integer greater than 0")
    #     if quadrature_order <= 0:
    #         raise ValueError("Quadrature order must be an integer greater than 0")
        
    #     w = np.linspace(-self.basis.wM, self.basis.wM, size)
    #     dw = 2*self.basis.wM/(size-1)
    #     kernel = (1j*self.__siw.wn[:,None]*np.pi/self.beta - w[None,:])**-1
    #     M = kernel * self.__quadrature_matrix(dw, size, quadrature_order)[None,-1,:]
    #     Mh = M.T.conjugate()
    #     return w, np.linalg.inv(Mh @ M) @ Mh @ self.Gloc.Fiw
    
    # def __quadrature_matrix(self, h, size, order):
    #     k = order
    #     col,row = np.meshgrid(*(np.arange(k+1),)*2)
    #     P = np.linalg.inv(row**col)
    #     a = -P[1,:]
    #     s = ((row**(col+1))/(col+1)) @ P
        
    #     S = np.zeros((size, size))
    #     S[k+1:,k+1:] = np.eye(size-k-1)
    #     S[:k+1,:k+1] = h*s
        
    #     A = np.zeros((size,size))
    #     A[:k+1,:k+1] = np.eye(k+1)
    #     for l in range(k+1,size):
    #         A[l,l-k:l+1] = np.flip(a)/h
        
    #     return np.linalg.inv(A) @ S



# def __screened_interaction(G, H):
#     if isinstance(G, MatsubaraGreen):
#         pi = pol_bubble_from_green(G)
#         what = zeroBubble(H, G.basis)
