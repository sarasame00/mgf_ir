import numpy as np
import sparse_ir as ir
from .lattice import Lattice



class MatsubaraAbstract:
    def __init__(self, ir_coefs, ir_basis):
        if ir_basis.size != ir_coefs.shape[0]:
            raise ValueError("Initialization coefficients are not coinciding with ir-basis size")
        self.__ir_basis = ir_basis
        self.__ir_coefs = ir_coefs
        self.__sampl_time = ir.TauSampling(ir_basis)
        self.__sampl_freq = ir.MatsubaraSampling(ir_basis)
    
    @property
    def beta(self):
        return self.__ir_basis.beta
    
    @property
    def dim(self):
        if self.__ir_coefs.ndim == 1:
            return 1
        return self.__ir_coefs.shape[-1]
    
    @property
    def particle(self):
        return self.basis.statistics
    
    @property
    def basis(self):
        return self.__ir_basis
    
    @property
    def Fl(self):
        return self.__ir_coefs
    
    @property
    def Ftau(self):
        return self.__sampl_time.evaluate(self.Fl, axis=0)
    
    @property
    def Fiw(self):
        return self.__sampl_freq.evaluate(self.Fl, axis=0)
    
    def __call__(self, r, space):
        if space=='time':
            poly = self.basis.u(r)
        elif space=='freq':
            poly = self.basis.uhat(r)
        else:
            raise ValueError("Space can be time or freq")
        
        if self.Fl.ndim == 2:
            return np.einsum('la,l...->...a', self.Fl, poly)
        elif self.Fl.ndim == 3:
            return np.einsum('lka,l...->...ka', self.Fl, poly)
        elif self.Fl.ndim == 4:
            return np.einsum('lka,l...->...ka', self.Fl, poly)
        else:
            raise ValueError("Bad dimension for Matsubara function ir coefficients")
    
    def __getitem__(self, k):
        return self.Fl[k]
    
    def __setitem__(self, k, val):
        self.__ir_coefs[k] = val
    
    def __add__(self, other):
        if type(self) != type(other):
            raise TypeError("Two different types of Matsubara functions cannot be operated")
        if self.basis != other.basis:
            raise ValueError("Only Matsubara functions with same ir-basis can be operated")
        return MatsubaraGreen(self[:] + other[:], self.basis)
    
    def __iadd__(self, other):
        if type(self) != type(other):
            raise TypeError("Two different types of Matsubara functions cannot be operated")
        if self.basis != other.basis:
            raise ValueError("Only Matsubara functions with same ir-basis can be operated")
        return MatsubaraGreen(self[:] + other[:], self.basis)
    
    def __sub__(self, other):
        if type(self) != type(other):
            raise TypeError("Two different types of Matsubara functions cannot be operated")
        if self.basis != other.basis:
            raise ValueError("Only Matsubara functions with same ir-basis can be operated")
        return MatsubaraGreen(self[:] - other[:], self.basis)
    
    def __isub__(self, other):
        if type(self) != type(other):
            raise TypeError("Two different types of Matsubara functions cannot be operated")
        if self.basis != other.basis:
            raise ValueError("Only Matsubara functions with same ir-basis can be operated")
        return MatsubaraGreen(self[:] - other[:], self.basis)
    
    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise ValueError("Matsubara Green's function can only be multiplied by a number")
        return MatsubaraGreen(self.__ir_coefs*other, self.basis)
    
    def __imul__(self, other):
        if not isinstance(other, (float, int)):
            raise ValueError("Matsubara Green's function can only be multiplied by a number")
        return MatsubaraGreen(self.__ir_coefs*other, self.basis)
    
    def __rmul__(self, other):
        if not isinstance(other, (float, int)):
            raise ValueError("Matsubara Green's function can only be multiplied by a number")
        return MatsubaraGreen(self.__ir_coefs*other, self.basis)


class MatsubaraAbstractLatt(MatsubaraAbstract):
    def __init__(self, ir_coefs, ir_basis, lattice):
        super().__init__(ir_coefs, ir_basis)
        if lattice.nk != ir_coefs.shape[1]:
            raise ValueError("Lattice Matsubara function must have at second axis the lattice information, shape not compatible with nk")
        self.__lattice = lattice
    
    @property
    def lattice(self):
        return self.__lattice
    
    @property
    def nk(self):
        return self.lattice.nk
    
    @property
    def Fl_real(self):
        return self.__to_real_space(self.Fl)
    
    @property
    def Ftau_real(self):
        return self.__to_real_space(self.Ftau)
    
    @property
    def Fiw_real(self):
        return self.__to_real_space(self.Fiw)
    
    def __to_real_space(self, Fk):
        euler = np.exp(-1j * np.einsum('ij...,k...->ijk',self.lattice.rij,self.lattice.k_vec))
        return np.sum(euler[None,:,:,:,None] * Fk[:,None,None,:,:], axis=3) / self.lattice.nk


class MatsubaraGreen(MatsubaraAbstract):
    def __init__(self, ir_coefs, ir_basis):
        if ir_coefs.ndim != 2:
            raise ValueError("Matsubara Green's function should have axes for ir-basis and quantum states")
        super().__init__(ir_coefs, ir_basis)


class MatsubaraGreenLatt(MatsubaraAbstractLatt):
    def __init__(self, ir_coefs, ir_basis, lattice):
        if ir_coefs.ndim != 3:
            raise ValueError("Lattice Matsubara Green's function should have axes for ir-basis, k vectros and quantum states")
        super().__init__(ir_coefs, ir_basis, lattice)


class MatsubaraBubble(MatsubaraAbstract):
    def __init__(self, ir_coefs, ir_basis):
        if ir_coefs.ndim != 3:
            raise ValueError("Matsubara bubble function should have an axis for ir-basis and two axes quantum states")
        super().__init__(ir_coefs, ir_basis)


class MatsubaraBubbleLatt(MatsubaraAbstractLatt):
    def __init__(self, ir_coefs, ir_basis, lattice):
        if ir_coefs.ndim != 4:
            raise ValueError("Lattice Matsubara Green's function should have axes for ir-basis, k vectros and two for quantum states")
        super().__init__(ir_coefs, ir_basis, lattice)


################################
# Constructors
################################

def non_int_g(H, mu, ir_basis, lattice=None):
    siw = ir.MatsubaraSampling(ir_basis)
    iw = 1j*siw.wn * np.pi/ir_basis.beta
    
    if isinstance(lattice, Lattice):
        hk = H.Hk(lattice) - mu
        giw = (iw[:,None,None] - hk[None,:,:])**-1
        return MatsubaraGreenLatt(siw.fit(giw, axis=0).real, ir_basis, lattice)
    
    else:
        h = H.w - mu
        giw = (iw[:,None] - h[None,:])**-1
        return MatsubaraGreen(siw.fit(giw, axis=0), ir_basis)


def zeroG(H, ir_basis, lattice=None):
    if isinstance(lattice, Lattice):
        return MatsubaraGreenLatt(np.zeros((ir_basis.size, lattice.nk, H.dim)), ir_basis, lattice)
    else:
        return MatsubaraGreen(np.zeros((ir_basis.size, H.dim)), ir_basis)


def mats_copy(F):
    if isinstance(F, MatsubaraGreen):
        return MatsubaraGreen(F.Fl, F.basis)
    elif isinstance(F, MatsubaraGreenLatt):
        return MatsubaraGreenLatt(F.Fl, F.basis, F.lattice)
    elif isinstance(F, MatsubaraBubble):
        return MatsubaraBubble(F.Fl, F.basis)
    elif isinstance(F, MatsubaraBubbleLatt):
        return MatsubaraBubbleLatt(F.Fl, F.basis, F.lattice)
    else:
        raise TypeError("This cannot copy an object which is not a Matsubara type")


def zeroBubble(H, ir_basis, lattice=None):
    if isinstance(lattice, Lattice):
        return MatsubaraBubbleLatt(np.zeros((ir_basis.size, lattice.nk, H.dim, H.dim)), ir_basis, lattice)
    else:
        return MatsubaraBubble(np.zeros((ir_basis.size, H.dim, H.dim)), ir_basis)


def pol_bubble_from_green(G):
    if not isinstance(G, (MatsubaraGreen, MatsubaraGreenLatt)):
        raise TypeError("Polarization bubble is constructed from Green's function")
    
    ps = {'F':1, 'B':1}[G.basis.statistics]
    stau = ir.TauSampling(G.basis)
    if isinstance(G, MatsubaraGreen):
        ir_coefs = stau.fit(ps*np.einsum('...a,...b,...ab', G.Ftau, np.flip(G.Ftau,axis=0)), axis=0)
        return MatsubaraBubble(ir_coefs, G.basis)
    
    elif isinstance(G, MatsubaraGreenLatt):
        pijtau = ps*np.einsum('...a,...b,...ab', G.Ftau_real*np.flip(G.Ftau_real, axis=0))
        rij = G.lattice.r_vec[None,:,:] - G.lattice.r_vec[:,None,:]
        euler = np.exp(1j * np.einsum('ij...,k...->ijk',rij,G.lattice.k_vec))
        pktau = np.sum(euler[None,:,:,:,None] * pijtau[:,:,:,None,:], axis=(1,2)) / G.nk
        ir_coefs = stau.fit(pktau, axis=0)
        return MatsubaraBubbleLatt(ir_coefs, G.basis)



# Self-energy approximations

def __self_energy_hf(H, Gloc):
    if Gloc.particle == 'F':
        ps = -1
    elif Gloc.particle == 'B':
        ps = 1
    
    sigma = ps * np.einsum('ab,lb,l->a', H.Umf, Gloc[:], Gloc.basis.u(Gloc.beta))
    siw = ir.MatsubaraSampling(Gloc.basis)
    
    return MatsubaraGreen(siw.fit(sigma[None,:] * np.ones((siw.wn.size,1)), axis=0).real, Gloc.basis)