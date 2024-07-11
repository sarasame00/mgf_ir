import numpy as np
import sparse_ir as ir
from .lattice import Lattice
from .hamiltonian import Hamiltonian
from .mats_func import non_int_g, zeroG, zeroBubble, pol_bubble_from_green, MatsubaraGreen, MatsubaraGreenLatt



class DysonSolver:
    def __init__(self, H, ir_basis, N, lattice=None, mu_init=None):
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
        
        if isinstance(mu_init, (int, float)):
            self.__mu = mu_init
        else:
            self.__mu = 0
        
        self.__G0 = non_int_g(H, self.__mu, ir_basis, lattice)
        self.__Gloc = zeroG(H, ir_basis)
        self.__Gk = zeroG(H, ir_basis, lattice) if self.__in_lattice else None
        self.__SE = zeroG(H, ir_basis)
        
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
    
    
    def __eval_particle_dens(self):
        return -np.sum(self.Gloc(self.beta, 'time'))
    
    def __update_self_energy(self, approx):
        ps = {'F':-1, 'B':1}[self.particle]
        sigma_hf = ps * np.einsum('ab,lb,l->a', self.H.Umf, self.Gloc[:], self.basis.u(self.beta))
        shfiw = sigma_hf * np.ones((self.__sampl_freq.wn.size, self.dim))
        shfl = self.__sampl_freq.fit(shfiw, axis=0).real
        
        if approx == "HF":
            self.__SE[:] = shfl
        
        elif approx == "GW":
            pik = pol_bubble_from_green(self.Gk)
            raise NotImplementedError
        
        else:
            raise NotImplementedError
    
    def __update_green(self):
        if self.__in_lattice:
            Gkiw = (self.g.Fiw**-1 - self.segy.Fiw[:,None,:])**-1
            self.__Gk[:] = self.__sampl_freq.fit(Gkiw, axis=0).real
            self.__Gloc[:] = np.sum(self.Gk[:], axis=1) / self.nk
        
        else:
            Giw = (self.g.Fiw**-1 - self.segy.Fiw)**-1
            self.__Gloc[:] = self.__sampl_freq.fit(Giw, axis=0).real
    
    def __checkN(self, N):
        return -np.sum(self.Gloc(self.beta, 'time'), axis=-1)
    
    def solve(self, approx = "HF", th=1e-6, delta_mu = 0.1, diis_mem=5):
        print("Starting self-consistent solution of Dyson equation\n")
        last_sign = 2
        while True: #Loop for mu
            loops = 0
            diis_err = np.zeros((diis_mem, self.basis.size, self.dim))
            diis_val = np.zeros((diis_mem, self.basis.size, self.dim))
            while True: #sc loop
                Glocl_prev = np.copy(self.Gloc.Fl)
                self.__update_self_energy(approx)
                self.__update_green()
                diis_val[:-1] = np.copy(diis_val[1:])
                diis_val[-1] = np.copy(self.Gloc.Fl)
                diis_err[:-1] = np.copy(diis_err[1:])
                diis_err[-1] = np.copy(self.Gloc.Fl - Glocl_prev)
                if loops >= diis_mem:
                    B = np.zeros((diis_mem,)*2)
                    for i in range(diis_mem):
                        for j in range(i, diis_mem):
                            B[i,j] = np.sum(diis_err[i] * diis_err[j])
                            if i != j:
                                B[j,i] = np.copy(B[i,j])
                    c_prime = np.linalg.inv(B) @ np.ones((diis_mem,))
                    c = c_prime / np.sum(c_prime)
                    self.__Gloc[:] = np.sum(c[:,None,None]*diis_val, axis=0)
                    diis_val[-1] = np.copy(self.Gloc.Fl)
                    diis_err[-1] = np.copy(self.Gloc.Fl - Glocl_prev)
                conv = np.sqrt(np.sum((Glocl_prev - self.Gloc.Fl)**2))
                loops += 1
                print("Iteration %i completed with convergence %.8f" % (loops, conv))
                if conv < th:
                    break
            DN = self.__N - self.__eval_particle_dens()
            print("Convergence aquired for mu=%.8f" % self.mu)
            print("Particle density computed: %.8f" % (self.__N - DN))
            
            if abs(DN) < th:
                break
            sign_N = 1 if DN > 0 else -1
            if not (last_sign == sign_N or last_sign == 2):
                delta_mu /= 2
            self.__mu += sign_N * delta_mu
            last_sign = sign_N
            
            print("New mu=%.8f" % self.__mu)
            print('-------------------------------------------------------\n')
            
            self.__G0 = non_int_g(self.H, self.mu, self.basis, self.lattice)
        print("Finished")
        self.__solved = True
    
    def spectral_function(self, w):
        if not self.__solved:
            raise AttributeError("Not able to give the spectral function before solve Dyson equation")
        
        rhol = -np.einsum('ij,lj,jk->lik', self.H.P, self.Gloc.Fl/self.basis.sl[:,None], self.H.Ph)
        poly = self.basis.v(w)
        return np.einsum('lij,l...->...ij', rhol, poly)



def __screened_interaction(G, H):
    if isinstance(G, MatsubaraGreen):
        pi = pol_bubble_from_green(G)
        what = zeroBubble(H, G.basis)