import numpy as np
import sparse_ir as ir
from .lattice import Lattice
from .hamiltonian import Hamiltonian
from .mats_func import non_int_g, zeroG, mats_copy, zeroBubble, pol_bubble_from_green, MatsubaraGreen, MatsubaraGreenLatt



class DysonSolver:
    def __init__(self, H, ir_basis, N, lattice=None):
        if not isinstance(H, Hamiltonian):
            raise TypeError("H must be a Hamiltonian type")
        self.__H = H
        
        self.__N = N
        
        self.__basis = ir_basis
        
        if isinstance(lattice, Lattice):
            self.__lattice = lattice
            self.__in_lattice = True
        else:
            self.__lattice = None
            self.__in_lattice = False
        
        
        self.__mu = 0
        
        self.__G0 = self.__non_int_mu_adj()
        self.__Gk = mats_copy(self.__G0) if self.__in_lattice else None
        self.__Gloc = MatsubaraGreen(np.sum(self.__Gk.Fl, axis=1)/self.nk, ir_basis) if self.__in_lattice else mats_copy(self.__G0)
        self.__SE = zeroG(H, ir_basis)
        self.__sehf = 0
        
        self.__sampl_time = ir.TauSampling(ir_basis)
        self.__sampl_freq = ir.MatsubaraSampling(ir_basis)
        
        
        self.__solved = False
    
    @property
    def H(self):
        return self.__H
    
    @property
    def particle_density(self):
        return self.__N
    
    @property
    def dim(self):
        return self.H.dim
    
    @property
    def basis(self):
        return self.__basis
    
    @property
    def beta(self):
        return self.basis.beta
    
    @property
    def particle(self):
        return self.basis.statistics
    
    @property
    def lattice(self):
        if self.__in_lattice:
            return self.__lattice
        else:
            raise AttributeError("This Dyson solver has no lattice")
    
    @property
    def nk(self):
        return self.lattice.nk
    
    @property
    def mu(self):
        return self.__mu
    
    @property
    def g(self):
        return self.__G0
    
    @property
    def Gloc(self):
        return self.__Gloc
    
    @property
    def Gk(self):
        if self.__in_lattice:
            return self.__Gk
        else:
            raise AttributeError("This Dyson solver hs no lattice")
    
    @property
    def segy(self):
        return self.__SE
    
    @property
    def segy_hf(self):
        return self.__sehf
    
    
    def __non_int_mu_adj(self, th=1e-8, delta_mu=0.5):
        last_sign = 2
        loops = 0
        print("Adjusting non-interacting chemichal potential")
        while True:
            g0 = non_int_g(self.H, self.mu, self.basis, self.lattice) if self.__in_lattice else non_int_g(self.H, self.mu, self.basis)
            if self.__in_lattice:
                N = -np.sum(g0.Ftau[-1]) / self.nk
            else:
                N = -np.sum(g0.Ftau[-1])
            DN = self.__N - N
            
            if abs(DN) < th:
                break
            sign_N = 1 if DN > 0 else -1
            if not (last_sign == sign_N or last_sign == 2):
                delta_mu /= 2
            loops += 1
            print("Iteration %i computed particle density of %.8f and mu=%.8f" % (loops, N, self.__mu))
            self.__mu += sign_N * delta_mu
            last_sign = sign_N
        print("Non interactive Green's function initialized with mu=%.8f" % self.mu)
        print('-------------------------------------------------------\n')
        return g0
    
    def __mu_adj(self, th=1e-6, delta_mu=0.5):
        self.__mu = 0
        last_sign = 2
        loops = 0
        print("Adjusting chemichal potential")
        while True:
            self.__G0 = non_int_g(self.H, self.mu, self.basis, self.lattice) if self.__in_lattice else non_int_g(self.H, self.mu, self.basis)
            self.__update_green()
            N = self.__eval_particle_dens()
            DN = self.__N - N
            
            if abs(DN) < th:
                break
            sign_N = 1 if DN > 0 else -1
            if not (last_sign == sign_N or last_sign == 2):
                delta_mu /= 2
            loops += 1
            print("Iteration %i computed particle density of %.8f and mu=%.8f" % (loops, N, self.__mu))
            self.__mu += sign_N * delta_mu
            last_sign = sign_N
        print("Green's function optimized with mu=%.8f" % self.mu)
    
    def __eval_particle_dens(self):
        return -np.sum(self.Gloc(self.beta, 'time'))
    
    def __update_self_energy(self, approx):
        ps = {'F':-1, 'B':1}[self.particle]
        self.__sehf = ps * self.H.Umf @ self.Gloc.Ftau[-1]
        
        if approx == "HF":
            pass
        
        elif approx == "GW":
            pik = pol_bubble_from_green(self.Gk)
            raise NotImplementedError
        
        else:
            raise NotImplementedError
    
    def __update_green(self):
        # ps = {'F':-1, 'B':1}[self.particle]
        # sigma_hf = ps * np.einsum('ab,lb,l->a', self.H.Umf, self.Gloc[:], self.basis.u(self.beta))
        
        if self.__in_lattice:
            Gkiw = (self.g.Fiw**-1 - self.segy_hf[None,None,:] - self.segy.Fiw[:,None,:])**-1
            self.__Gk[:] = self.__sampl_freq.fit(Gkiw, axis=0).real
            self.__Gloc[:] = np.sum(self.Gk[:], axis=1) / self.nk
        
        else:
            Giw = (self.g.Fiw**-1 - self.segy.Fiw)**-1
            self.__Gloc[:] = self.__sampl_freq.fit(Giw, axis=0).real
    
    def __checkN(self, N):
        return -np.sum(self.Gloc.Ftau[-1])
        # return -np.sum(self.Gloc(self.beta, 'time'), axis=-1)
    
    
    def solve(self, approx = "HF", th=1e-6, diis_mem=5):
        print("Starting self-consistent solution of Dyson equation\n")
        loops = 0
        diis_err = np.zeros((diis_mem, self.basis.size+1, self.dim))
        diis_val = np.zeros((diis_mem, self.basis.size+1, self.dim))
        while True:
            last_Glocl = np.copy(self.Gloc.Fl)
            self.__update_self_energy(approx)
            diis_vec = np.zeros((self.basis.size+1,self.dim))
            diis_vec[0] = np.copy(self.segy_hf)
            diis_vec[1:] = np.copy(self.segy.Fl)
            diis_val[:-1] = np.copy(diis_val[1:])
            diis_val[-1] = np.copy(diis_vec)
            diis_err[:-1] = np.copy(diis_err[1:])
            diis_err[-1] = np.copy(diis_val[-1] - diis_val[-2])
            if loops >= diis_mem:
                B = np.zeros((diis_mem,)*2)
                for i in range(diis_mem):
                    for j in range(i, diis_mem):
                        B[i,j] = np.sum(diis_err[i] * diis_err[j])
                        if i != j:
                            B[j,i] = np.copy(B[i,j])
                c_prime = np.linalg.inv(B) @ np.ones((diis_mem,))
                c = c_prime / np.sum(c_prime)
                diis_vec = np.sum(c[:,None,None]*diis_val, axis=0)
                diis_val[-1] = np.copy(diis_vec)
                diis_err[-1] = np.copy(diis_val[-1] - diis_val[-2])
                self.__segy = np.copy(diis_vec[0])
                self.__SE[:] = np.copy(diis_vec[1:])
            
            self.__mu_adj()
            
            conv = np.sqrt(np.sum((self.Gloc.Fl - last_Glocl)**2))
            loops += 1
            if conv < th:
                break
            print("Iteration %i completed with convergence %.8f" % (loops, conv))
            print('-------------------------------------------------------\n')
        print("Finished after %i iterations with convergence of %.8f" % (loops, conv))
        self.__solved = True
    
    def spectral_function_mf(self):
        def rho(w, eta):
            if not isinstance(eta, (int, float)):
                raise ValueError("Dumping must be a constant number")
            
            if isinstance(w, (int, float)):
                if w < -self.basis.wmax or w > self.basis.wmax:
                    raise ValueError("Not able to compute outside max frequency %.3feV" % self.basis.wmax)
                if self.__in_lattice:
                    A = eta/self.nk/np.pi * np.sum(((w - self.H.Hk(self.lattice) - self.segy_hf[None,:] + self.mu)**2 + eta**2)**-1, axis=0)
                else:
                    A = eta/np.pi * ((w - self.H.w - self.segy_hf + self.mu)**2 + eta**2)**-1
            
            else:
                try:
                    w = np.array(w)
                except:
                    raise TypeError("Frequencies must be an iterable of numbers")
                if np.any(w < -self.basis.wmax) or np.any(w > self.basis.wmax):
                    raise ValueError("Not able to compute outside max frequency %.3feV" % self.basis.wmax)
                if self.__in_lattice:
                    A = eta/self.nk/np.pi * np.sum(((w[:,None,None] - self.H.Hk(self.lattice)[None,:,:] - self.segy_hf[None,None,:] + self.mu)**2 + eta**2)**-1, axis=1)
                else:
                    A = eta/np.pi * ((w[:,None] - self.H.w[None,:] - self.segy_hf[None,:] + self.mu)**2 + eta**2)**-1
            
            return np.einsum('ij,...j,jk->...ik', self.H.P, A, self.H.Ph)
        
        return rho

    def particle_number(self):
        N = self.Gloc.Ftau[-1]
        return np.einsum('ij,j,jk->ik', self.H.P, N, self.H.Ph)

    def to_eigenbasis(self, A):
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



def __screened_interaction(G, H):
    if isinstance(G, MatsubaraGreen):
        pi = pol_bubble_from_green(G)
        what = zeroBubble(H, G.basis)
