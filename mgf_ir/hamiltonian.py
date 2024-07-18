import numpy as np



class Hamiltonian:
    def __init__(self, Hloc, V, Hkin=None):
        if Hloc.ndim != 2:
            raise ValueError("The local Hamiltonian is a hermitian matrix")
        if Hloc.shape[0] != Hloc.shape[1]:
            raise ValueError("The local Hamiltonian is a hermitian matrix")
        if not np.all(np.isclose(Hloc.T.conjugate(), Hloc)):
            raise ValueError("The local Hamiltonian is a hermitian matrix")
        if isinstance(Hkin, np.ndarray):
            if Hkin.ndim != 3:
                raise ValueError("Kinetic Hamiltinian must contain three hermitian matrices, one for every cartesian direction")
            if Hkin.shape[0] != 3 and Hkin.shape[1] != Hkin.shape[2]:
                raise ValueError("Kinetic Hamiltinian must contain three hermitian matrices, one for every cartesian direction")
            if Hkin.shape[0] != Hkin.shape[0]:
                raise ValueError("Local and kinetic Hamiltonians must have the same Hilbert dimension")
            if not np.all(np.isclose(np.swapaxes(Hkin, 1, 2).conjugate(), Hkin)):
                raise ValueError("Kinetic Hamiltinian must contain three hermitian matrices, one for every cartesian direction")
        if V.shape[0] != Hloc.shape[0] or V.shape[1] != Hloc.shape[0] or V.shape[2] != Hloc.shape[0] or V.shape[3] != Hloc.shape[0]:
            raise ValueError("Interaction tensor must have the same dimension that local Hamiltonian")
        
        self.__dim = Hloc.shape[0]
        
        self.__w, self.__P = np.linalg.eigh(Hloc)
        self.__hopp = -np.einsum('ik,...kl,li->...i', self.Ph, Hkin, self.P).real if isinstance(Hkin, np.ndarray) else None
        self.__Umf = -np.einsum('bk,am,na,lb,mkln->ab', self.Ph, self.Ph, self.P, self.P, V - np.swapaxes(V, 2, 3)).real
        self.__U = np.einsum('bk,am,na,ld,mkln->ab', self.Ph, self.Ph, self.P, self.P, V).real
    
    @property
    def dim(self):
        return self.__dim
    
    @property
    def w(self):
        return self.__w
    
    @property
    def P(self):
        return self.__P
    
    @property
    def Ph(self):
        return self.__P.T.conjugate()
    
    @property
    def kin_hoppings(self):
        if self.__hopp is None:
            raise AttributeError("This Hamiltonian object has not a kietic term")
        return self.__hopp
    
    @property
    def Umf(self):
        return self.__Umf
    
    @property
    def U(self):
        return self.__U
    
    def Hk(self, lattice):
        if self.__hopp is None:
            raise AttributeError("This Hamiltonian object has not a kietic term")
        return self.w[None,:] + 2*np.sum(self.kin_hoppings[None,:,:] * np.cos(lattice.k_vecs)[:,:,None], axis=1)