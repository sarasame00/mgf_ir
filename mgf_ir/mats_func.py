import numpy as np
import sparse_ir as ir
from .lattice import Lattice



class MatsubaraAbstract:
    # This class handles computations related to Matsubara frequencies and uses sparse representation and sampling of IR coefficients.

    def __init__(self, ir_coefs, ir_basis):
        """
        Initializes the MatsubaraAbstract class with IR coefficients and IR basis.
        It also initializes time and frequency samplers.
        """

        if ir_basis.size != ir_coefs.shape[0]:
            # Check that the size of the IR basis matches the number of IR coefficients
            raise ValueError("Initialization coefficients are not coinciding with ir-basis size")
        
        self.__ir_basis = ir_basis # Assign the IR basis to the object (defines the grid for Matsubara frequencies or times)
        self.__ir_coefs = ir_coefs  # Assign the IR coefficients (used in calculations)
        
        # Initialize TauSampling and MatsubaraSampling objects using the IR basis
        self.__ir_coefs = ir_coefs
        self.__sampl_time = ir.TauSampling(ir_basis)
        self.__sampl_freq = ir.MatsubaraSampling(ir_basis)
    
    @property
    def beta(self):
        # This property calculates and returns the Matsubara beta value based on the IR basis.
        return self.__ir_basis.beta
    
    @property
    def dim(self):
        # This property calculates the dimensionality of the IR coefficients.

        if self.__ir_coefs.ndim == 1:
            return 1
        return self.__ir_coefs.shape[-1]
    
    @property
    def particle(self):
        # This property returns the statistics of the basis.
        return self.basis.statistics
    
    @property
    def basis(self):
        # This property returns the IR basis used for the Matsubara function.
        return self.__ir_basis
    
    @property
    def Fl(self):
        # This property returns the IR coefficients (Fl) for the Matsubara function.

        return self.__ir_coefs
    
    @property
    def Ftau(self):
        """
        This property returns the Matsubara function evaluated in the time domain (Tau).
        It uses the time sampler (__sampl_time) to evaluate the IR coefficients (Fl) along the time axis.
        """
        return self.__sampl_time.evaluate(self.Fl, axis=0)
    
    @property
    def Fiw(self):
        """
        This property returns the Matsubara function evaluated in the frequency domain (Iw).
        It uses the frequency sampler (__sampl_freq) to evaluate the IR coefficients (Fl) along the frequency axis.
        """
        return self.__sampl_freq.evaluate(self.Fl, axis=0)
    
    def __call__(self, r, space):
        """
        This method allows the object to be called as a function.
        It takes two arguments: r (the position or index) and space (which can be either 'time' or 'freq').
        Depending on the space, it computes the appropriate quantity using the basis' time or frequency representation.
        """
        if space=='time':
            poly = self.basis.u(r)
        elif space=='freq':
            poly = self.basis.uhat(r)
        else:
            raise ValueError("Space can be time or freq")
        
        # Perform a matrix-vector multiplication (or tensor contraction) depending on the dimensions of the IR coefficients
        if self.Fl.ndim == 2:
            return np.einsum('la,l...->...a', self.Fl, poly)
        elif self.Fl.ndim == 3:
            return np.einsum('lka,l...->...ka', self.Fl, poly)
        elif self.Fl.ndim == 4:
            return np.einsum('lka,l...->...ka', self.Fl, poly)
        else:
            raise ValueError("Bad dimension for Matsubara function ir coefficients")
    
    def __getitem__(self, k):
        """
        This method allows indexing into the Matsubara function (e.g., self[k]) to access specific coefficients.
        """
        return self.Fl[k]
    
    def __setitem__(self, k, val):
        """
        This method allows setting a value for specific coefficients in the Matsubara function.
        """
        self.__ir_coefs[k] = val
    
    def __add__(self, other):
        """
        This method defines the behavior of the addition operator (+) for two Matsubara functions.
        It checks if both operands are of the same type and have the same IR basis, then performs the addition.
        """
        if type(self) != type(other):
            raise TypeError("Two different types of Matsubara functions cannot be operated")
        if self.basis != other.basis:
            raise ValueError("Only Matsubara functions with same ir-basis can be operated")
        return MatsubaraGreen(self[:] + other[:], self.basis)
    
    def __iadd__(self, other):
        """
        This method defines the behavior of the in-place addition operator (+=) for two Matsubara functions.
        """
        if type(self) != type(other):
            raise TypeError("Two different types of Matsubara functions cannot be operated")
        if self.basis != other.basis:
            raise ValueError("Only Matsubara functions with same ir-basis can be operated")
        return MatsubaraGreen(self[:] + other[:], self.basis)
    
    def __sub__(self, other):
        """
        This method defines the behavior of the subtraction operator (-) for two Matsubara functions.
        It checks if both operands are of the same type and have the same IR basis, then performs the subtraction.
        """
        if type(self) != type(other):
            raise TypeError("Two different types of Matsubara functions cannot be operated")
        if self.basis != other.basis:
            raise ValueError("Only Matsubara functions with same ir-basis can be operated")
        return MatsubaraGreen(self[:] - other[:], self.basis)
    
    def __isub__(self, other):
        """
        This method defines the behavior of the in-place subtraction operator (-=) for two Matsubara functions.
        """
        if type(self) != type(other):
            raise TypeError("Two different types of Matsubara functions cannot be operated")
        if self.basis != other.basis:
            raise ValueError("Only Matsubara functions with same ir-basis can be operated")
        return MatsubaraGreen(self[:] - other[:], self.basis)
    
    def __mul__(self, other):
        """
        This method defines the behavior of the multiplication operator (*) for a Matsubara function and a scalar (float or int).
        It multiplies the IR coefficients by the scalar value.
        """
        if not isinstance(other, (float, int)):
            raise ValueError("Matsubara Green's function can only be multiplied by a number")
        return MatsubaraGreen(self.__ir_coefs*other, self.basis)
    
    def __imul__(self, other):
        """
        This method defines the behavior of the in-place multiplication operator (*=) for a Matsubara function and a scalar.
        It multiplies the IR coefficients by the scalar value in-place.
        """
        if not isinstance(other, (float, int)):
            raise ValueError("Matsubara Green's function can only be multiplied by a number")
        return MatsubaraGreen(self.__ir_coefs*other, self.basis)
    
    def __rmul__(self, other):
        """
        This method defines the behavior of the reverse multiplication operator for a Matsubara function and a scalar.
        """
        if not isinstance(other, (float, int)):
            raise ValueError("Matsubara Green's function can only be multiplied by a number")
        return MatsubaraGreen(self.__ir_coefs*other, self.basis)


class MatsubaraAbstractLatt(MatsubaraAbstract):
    # Initialize the MatsubaraAbstractLatt class, inheriting from MatsubaraAbstract

    def __init__(self, ir_coefs, ir_basis, lattice):

        # Call the constructor of the base class MatsubaraAbstract
        super().__init__(ir_coefs, ir_basis)

        # Check that the lattice's nk matches the second axis of ir_coefs (for dimensional compatibility)
        if lattice.nk != ir_coefs.shape[1]:
            raise ValueError("Lattice Matsubara function must have at second axis the lattice information, shape not compatible with nk")
        
        # Store the lattice object in the class instance
        self.__lattice = lattice
    
    @property
    def lattice(self):
        """
        This property returns the lattice object that was provided during initialization.
        The lattice contains information about the grid and lattice vectors.
        """
        return self.__lattice
    
    @property
    def nk(self):
        """
        This property returns the number of lattice points (nk) from the lattice object.
        The number of lattice points is used to handle the sampling and evaluation of the Matsubara function.
        """
        return self.lattice.nk
    
    @property
    def Fl_real(self):
        """
        This property converts the Matsubara function (Fl) to real space by applying the transformation.
        It calls the private method __to_real_space to handle the conversion.
        """
        return self.__to_real_space(self.Fl)
    
    @property
    def Ftau_real(self):
        """
        This property converts the Matsubara function (Ftau) to real space, similar to Fl_real.
        """
        return self.__to_real_space(self.Ftau)
    
    @property
    def Fiw_real(self):
        """
        This property converts the Matsubara function (Fiw) to real space, similar to Fl_real.
        """
        return self.__to_real_space(self.Fiw)
    
    def __to_real_space(self, Fk):
        """
        This method transforms a Matsubara function (Fk) from momentum space to real space.
        It involves a sum over the lattice vectors and their corresponding exponential terms.
        """
        euler = np.exp(-1j * np.einsum('ij...,k...->ijk',self.lattice.rij,self.lattice.k_vec)) # Exponential term for real space transformation
        # Perform a summation over the appropriate axes to transform the function to real space
        return np.sum(euler[None, :, :, :, None] * Fk[:, None, None, :, :], axis=3) / self.lattice.nk  # Normalize by the number of lattice points


class MatsubaraGreen(MatsubaraAbstract):
    """
    Initializes the MatsubaraGreen class, which represents the Green's function in Matsubara frequencies.
    """
    def __init__(self, ir_coefs, ir_basis):
        if ir_coefs.ndim != 2:
            raise ValueError("Matsubara Green's function should have axes for ir-basis and quantum states")
        super().__init__(ir_coefs, ir_basis) # Call the constructor of MatsubaraAbstract


class MatsubaraGreenLatt(MatsubaraAbstractLatt):
    """
    Initializes the MatsubaraGreenLatt class, which represents the Matsubara Green's function with lattice information.
    """
    def __init__(self, ir_coefs, ir_basis, lattice):
        if ir_coefs.ndim != 3:
            raise ValueError("Lattice Matsubara Green's function should have axes for ir-basis, k vectros and quantum states")
        super().__init__(ir_coefs, ir_basis, lattice) # Call the constructor of MatsubaraAbstractLatt


class MatsubaraBubble(MatsubaraAbstract):
    """
    Initializes the MatsubaraBubble class, which represents the Matsubara bubble function.
    """
    def __init__(self, ir_coefs, ir_basis):
        if ir_coefs.ndim != 3:
            raise ValueError("Matsubara bubble function should have an axis for ir-basis and two axes quantum states")
        super().__init__(ir_coefs, ir_basis) # Call the constructor of MatsubaraAbstract


class MatsubaraBubbleLatt(MatsubaraAbstractLatt):
    """
    Initializes the MatsubaraBubbleLatt class, which represents the Matsubara bubble function with lattice information.
    """
    def __init__(self, ir_coefs, ir_basis, lattice):
        if ir_coefs.ndim != 4:
            raise ValueError("Lattice Matsubara Green's function should have axes for ir-basis, k vectros and two for quantum states")
        super().__init__(ir_coefs, ir_basis, lattice) # Call the constructor of MatsubaraAbstractLatt


################################
# Constructors
################################

def non_int_g(H, mu, ir_basis, lattice=None):
    """
    Constructs the Matsubara Green's function (non-interacting) for a given Hamiltonian H and chemical potential mu.
    This function considers whether a lattice is provided or not.
    """
    siw = ir.MatsubaraSampling(ir_basis) # Create a Matsubara sampling object for the given IR basis
    iw = 1j*siw.wn * np.pi/ir_basis.beta # Calculate the Matsubara frequencies (iω_n)
    
    if isinstance(lattice, Lattice): # If a lattice is provided
        hk = H.Hk(lattice) - mu # Compute the Hamiltonian in momentum space minus the chemical potential
        giw = (iw[:,None,None] - hk[None,:,:])**-1 # Compute the Green's function in the Matsubara frequency domain
        return MatsubaraGreenLatt(siw.fit(giw, axis=0).real, ir_basis, lattice) # Return the Matsubara Green's function for lattices
    
    else: # If no lattice is provided
        h = H.w - mu # Compute the Hamiltonian in frequency space minus the chemical potential
        giw = (iw[:,None] - h[None,:])**-1 # Compute the Green's function for non-lattice cases
        return MatsubaraGreen(siw.fit(giw, axis=0), ir_basis) # Return the Matsubara Green's function without lattice


def zeroG(H, ir_basis, lattice=None):
    """
    Returns a zero Matsubara Green's function, either with or without lattice, depending on the input.
    """
    if isinstance(lattice, Lattice):
        return MatsubaraGreenLatt(np.zeros((ir_basis.size, lattice.nk, H.dim)), ir_basis, lattice)
    else:
        return MatsubaraGreen(np.zeros((ir_basis.size, H.dim)), ir_basis)


def mats_copy(F):
    """
    Copies a Matsubara function object, whether it is MatsubaraGreen, MatsubaraGreenLatt, MatsubaraBubble, or MatsubaraBubbleLatt.
    """
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
    """
    Returns a zero Matsubara bubble function, either with or without lattice, depending on the input.
    """
    if isinstance(lattice, Lattice):
        return MatsubaraBubbleLatt(np.zeros((ir_basis.size, lattice.nk, H.dim, H.dim)), ir_basis, lattice)
    else:
        return MatsubaraBubble(np.zeros((ir_basis.size, H.dim, H.dim)), ir_basis)


def pol_bubble_from_green(G):
    """
    Constructs a polarization bubble from a given Matsubara Green's function (G).
    """
    if not isinstance(G, (MatsubaraGreen, MatsubaraGreenLatt)):
        raise TypeError("Polarization bubble is constructed from Green's function")
    
    ps = {'F':1, 'B':1}[G.basis.statistics] # Get the particle statistics (Fermions or Bosons)
    stau = ir.TauSampling(G.basis)  # Create a TauSampling object for the basis
    
    if isinstance(G, MatsubaraGreen):
        ir_coefs = stau.fit(ps*np.einsum('...a,...b,...ab', G.Ftau, np.flip(G.Ftau,axis=0)), axis=0) # Compute polarization bubble in the absence of lattice
        return MatsubaraBubble(ir_coefs, G.basis)  # Return Matsubara bubble without lattice
    
    elif isinstance(G, MatsubaraGreenLatt):
        pijtau = ps*np.einsum('...a,...b,...ab', G.Ftau_real*np.flip(G.Ftau_real, axis=0)) # Compute polarization bubble for lattice
        rij = G.lattice.r_vec[None,:,:] - G.lattice.r_vec[:,None,:] # Calculate pairwise lattice vectors
        euler = np.exp(1j * np.einsum('ij...,k...->ijk',rij,G.lattice.k_vec)) # Exponential factor for momentum space
        pktau = np.sum(euler[None,:,:,:,None] * pijtau[:,:,:,None,:], axis=(1,2)) / G.nk # Summing over the lattice and normalizing
        ir_coefs = stau.fit(pktau, axis=0) # Fit the polarization bubble
        return MatsubaraBubbleLatt(ir_coefs, G.basis) # Return Matsubara bubble with lattice



# Self-energy approximations

def __self_energy_hf(H, Gloc):
    if Gloc.particle == 'F':
        ps = -1
    elif Gloc.particle == 'B':
        ps = 1
    
    sigma = ps * np.einsum('ab,lb,l->a', H.Umf, Gloc[:], Gloc.basis.u(Gloc.beta))
    siw = ir.MatsubaraSampling(Gloc.basis)
    
    return MatsubaraGreen(siw.fit(sigma[None,:] * np.ones((siw.wn.size,1)), axis=0).real, Gloc.basis)

def __self_energy_hf(H, Gloc):
    """
    This function calculates the self-energy for the system using the Hubbard-Fermi mean-field approximation.
    It uses the Umf matrix (H.Umf) and the Green's function (Gloc).
    """
    if Gloc.particle == 'F':
        ps = -1  # For fermions, the self-energy has a sign of -1 (due to fermionic statistics)
    elif Gloc.particle == 'B':
        ps = 1  # For bosons, the self-energy has a sign of +1 (due to bosonic statistics)
    
    # Calculate the self-energy using an Einstein summation, which is shorthand for multiple summations over indices
    # H.Umf: Interaction matrix 
    # Gloc[:]: Local Green's function (Gloc is the Green's function for the system)
    # Gloc.basis.u(Gloc.beta): The basis function in the time domain
    sigma = ps * np.einsum('ab,lb,l->a', H.Umf, Gloc[:], Gloc.basis.u(Gloc.beta))  
    
    # Perform Matsubara sampling (sample on the Matsubara frequencies) for the basis
    siw = ir.MatsubaraSampling(Gloc.basis)  # Create a Matsubara sampling object for the given basis
    
    # Fit the computed self-energy (sigma) using the Matsubara sampling and return it as a MatsubaraGreen object
    return MatsubaraGreen(siw.fit(sigma[None, :] * np.ones((siw.wn.size, 1)), axis=0).real, Gloc.basis)
