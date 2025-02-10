import numpy as np


class Hamiltonian:
    
    def __init__(self, Hloc, V, Hkin=None):
        """
        The constructor initializes the Hamiltonian object. It takes the local Hamiltonian (Hloc), interaction tensor (V),
        and an optional kinetic Hamiltonian (Hkin) for a lattice system. The constructor checks various conditions
        to ensure the input matrices are consistent with the expected structure.
        """
        # Validate the local Hamiltonian matrix Hloc
        if Hloc.ndim != 2:
            raise ValueError("The local Hamiltonian is a hermitian matrix")
        if Hloc.shape[0] != Hloc.shape[1]:
            raise ValueError("The local Hamiltonian is a hermitian matrix")
        if not np.all(np.isclose(Hloc.T.conjugate(), Hloc)):
            raise ValueError("The local Hamiltonian is a hermitian matrix")
        
        # Validate the kinetic Hamiltonian matrix Hkin (if provided)
        if isinstance(Hkin, np.ndarray):
            if Hkin.ndim != 3:
                raise ValueError("Kinetic Hamiltinian must contain three hermitian matrices, one for every cartesian direction")
            if Hkin.shape[0] != 3 and Hkin.shape[1] != Hkin.shape[2]:
                raise ValueError("Kinetic Hamiltinian must contain three hermitian matrices, one for every cartesian direction")
            if Hkin.shape[0] != Hkin.shape[0]:
                raise ValueError("Local and kinetic Hamiltonians must have the same Hilbert dimension")
            if not np.all(np.isclose(np.swapaxes(Hkin, 1, 2).conjugate(), Hkin)):
                raise ValueError("Kinetic Hamiltinian must contain three hermitian matrices, one for every cartesian direction")
            
        # Validate the interaction tensor V
        if V.shape[0] != Hloc.shape[0] or V.shape[1] != Hloc.shape[0] or V.shape[2] != Hloc.shape[0] or V.shape[3] != Hloc.shape[0]:
            raise ValueError("Interaction tensor must have the same dimension that local Hamiltonian")
        
        # Store the dimensionality of the local Hamiltonian
        self.__dim = Hloc.shape[0]
        
        # Perform eigenvalue decomposition of the local Hamiltonian Hloc
        self.__w, self.__P = np.linalg.eigh(Hloc)

        # If a kinetic Hamiltonian (Hkin) is provided, compute the hopping terms
        self.__hopp = -np.einsum('ik,...kl,li->...i', self.Ph, Hkin, self.P).real if isinstance(Hkin, np.ndarray) else None
        
        # Compute the interaction terms Umf and U using the interaction tensor V
        self.__Umf = -np.einsum('bk,am,na,lb,mkln->ab', self.Ph, self.Ph, self.P, self.P, V - np.swapaxes(V, 2, 3)).real
        self.__U = np.einsum('bk,am,na,ld,mkln->ab', self.Ph, self.Ph, self.P, self.P, V).real
    
    @property
    def dim(self):
        """
        Returns the dimensionality of the Hamiltonian, which corresponds to the size of the Hilbert space.
        """
        return self.__dim
    
    @property
    def w(self):
        """
        Returns the eigenvalues of the local Hamiltonian (Hloc) from the eigenvalue decomposition.
        """
        return self.__w
    
    @property
    def P(self):
        """
        Returns the eigenvectors (P) of the local Hamiltonian (Hloc), which are used to diagonalize the Hamiltonian.
        """
        return self.__P
    
    @property
    def Ph(self):
        """
        Returns the conjugate transpose of the eigenvectors (P) of the local Hamiltonian.
        This is used to compute the interaction terms and hopping terms.
        """
        return self.__P.T.conjugate()
    
    @property
    def kin_hoppings(self):
        """
        Returns the kinetic hopping terms. Raises an error if the Hamiltonian does not include a kinetic term (Hkin).
        """
        if self.__hopp is None:
            raise AttributeError("This Hamiltonian object has not a kietic term")
        return self.__hopp
    
    @property
    def Umf(self):
        """
        Returns the mean-field interaction term (Umf), which is computed using the interaction tensor (V).
        """
        return self.__Umf
    
    @property
    def U(self):
        """
        Returns the interaction term (U), which is computed using the interaction tensor (V).
        """
        return self.__U
    
    def Hk(self, lattice):
        """
        Returns the Hamiltonian in momentum space (Hk), which is computed using the hopping terms and the lattice's k-vectors.
        This method calculates the Hamiltonian for the lattice system, including both the eigenvalues and the hopping terms.
        """
        if self.__hopp is None:
            raise AttributeError("This Hamiltonian object has not a kietic term")
        # Calculate the Hamiltonian in momentum space, including hopping terms
        return self.w[None,:] + 2*np.sum(self.kin_hoppings[None,:,:] * np.cos(lattice.k_vecs)[:,:,None], axis=1)