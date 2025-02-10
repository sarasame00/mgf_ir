import numpy as np


class Lattice:
    # Defining a class 'Lattice' that represents a lattice structure.
    # The class constructor initializes a lattice object with certain parameters.

    def __init__(self, nk1, nk2, nk3, basis_kind="cubic", a=None, b=None, c=None):
        """
        The constructor initializes the lattice structure with three main parameters (nk1, nk2, nk3) 
        that define the grid of points in the lattice. Optionally, it takes the type of lattice (basis_kind)
        and some parameters (a, b, c) that can be used to define specific properties of the lattice.
        """

        self.__nk1 = nk1  # Set the number of points along the first dimension (k1)
        self.__nk2 = nk2  # Set the number of points along the second dimension (k2)
        self.__nk3 = nk3  # Set the number of points along the third dimension (k3)
        
        # Call the __basis_set function to determine the lattice basis (in real and reciprocal space)
        # and assign the results to the instance variables __r_basis and __k_basis.
        self.__r_basis, self.__k_basis = self.__basis_set(basis_kind, a, b, c)
        
        # Calculate the real-space lattice vectors (__r_vecs) based on the basis and grid points.
        # Using numpy's einsum function for efficient computation of matrix multiplication.
        # __r_basis represents the lattice basis in real space, and np.meshgrid generates a grid of points in 3D.
        self.__r_vecs = np.einsum('ij,...j->...i', self.__r_basis, np.swapaxes(np.array(np.meshgrid(np.arange(nk1), np.arange(nk2), np.arange(nk3))), 1,2).reshape(3,nk1*nk2*nk3).T)
        
        # Calculate the k-space lattice vectors (__k_vecs) using the k_basis and the mesh grid of points
        self.__k_vecs = np.einsum('ij,...j->...i', self.__k_basis, np.swapaxes(np.array(np.meshgrid(np.arange(nk1), np.arange(nk2), np.arange(nk3))), 1,2).reshape(3,nk1*nk2*nk3).T/np.array([nk1,nk2,nk3]).reshape((1,3)))
        # self.__k_vecs = np.swapaxes(np.swapaxes(np.array(np.meshgrid(np.arange(nk1),np.arange(nk2),np.arange(nk3))), 1, 2).reshape((3,nk1*nk2*nk3)), 0, 1) * 2*np.pi/np.array([nk1,nk2,nk3]).reshape((1,3))
    
    @property
    def nk(self):
        # Return the total number of lattice points in the grid (nk1 * nk2 * nk3)
        return self.__nk1 * self.__nk2 * self.__nk3
    
    @property
    def a1(self):
        # Return the first lattice vector in real space (first column of r_basis)
        return self.__r_basis[:,0]
    
    @property
    def a2(self):
        # Return the second lattice vector in real space (second column of r_basis)
        return self.__r_basis[:,1]
    
    @property
    def a3(self):
        # Return the third lattice vector in real space (third column of r_basis)
        return self.__r_basis[:,2]
    
    @property
    def b1(self):
        # Return the first lattice vector in reciprocal space (first column of k_basis)
        return self.__k_basis[:,0]
    
    @property
    def b2(self):
        # Return the second lattice vector in reciprocal space (second column of k_basis)
        return self.__k_basis[:,1]
    
    @property
    def b3(self):
        # Return the third lattice vector in reciprocal space (third column of k_basis)
        return self.__k_basis[:,2]
    
    @property
    def r_vecs(self):
        # Return the real-space lattice vectors (__r_vecs)
        return self.__r_vecs
    
    @property
    def rij(self):
        # Return the pairwise differences between all lattice points (Rj - Ri)
        return self.r_vecs[None,:,:] - self.r_vecs[:,None,:]
    
    @property
    def k_vecs(self):
        # Return the reciprocal space lattice vectors (__k_vecs)
        return self.__k_vecs
    

    def __basis_set(self, kind, a, b, c):
        # This function generates the lattice basis set (real and reciprocal) based on the lattice type.

        if kind=="cubic":
            # If the lattice is cubic, the real-space basis vectors are just the unit vectors
            r_basis = np.eye(3)
        else:
            # For now, raise an error if the lattice kind is not implemented (only cubic is supported)
            raise NotImplementedError()
        
        # Initialize the reciprocal space basis vectors (k_basis)
        k_basis = np.zeros_like(r_basis)

        # Calculate the volume of the unit cell using the cross product of the real-space basis vectors
        vol = np.dot(np.cross(r_basis[:,0],r_basis[:,1]),r_basis[:,2])

        # Calculate the reciprocal space basis vectors based on the real-space basis
        for i in range(3):
            j = (i+1)%3
            k = (i-1)%3
            k_basis[:,i] = 2*np.pi * np.cross(r_basis[:,j],r_basis[:,k]) / vol
        
        # Return both real and reciprocal lattice basis vectors
        return r_basis, k_basis
    

    def dft(self, A, axis=-1):
        # Perform the discrete Fourier transform (DFT) of a given array A along the specified axis

        if A.shape[axis] != self.nk:
            # Check that the array's shape matches the size of the lattice
            raise ValueError("Axis of transformation must have the same size as lattice")
        
        if axis < 0:
             # Adjust the axis if it's negative (Python allows negative indices)
            axis += A.ndim
        
        # Calculate the dot product between real-space and reciprocal-space vectors
        r_dot_k = np.einsum('ri,ki->rk', self.r_vecs, self.k_vecs).reshape((1,)*axis + (self.nk,)*2 + (1,)*(A.ndim-1-axis))

        # Perform the DFT using the formula for a periodic lattice, summing over the lattice points
        return np.sum(np.exp(1j*r_dot_k) * A.reshape(A.shape[:axis+1] + (1,) + A.shape[axis+1:]), axis) / np.sqrt(self.nk)
    

    def idft(self, A, axis=-1):
        # Perform the inverse discrete Fourier transform (IDFT) of a given array A along the specified axis

        if A.shape[axis] != self.nk:
            # Check that the array's shape matches the size of the lattice
            raise ValueError("Axis of transformation must have the same size as lattice")
        
        if axis < 0:
            # Adjust the axis if it's negative (Python allows negative indices)
            axis += A.ndim

        # Calculate the dot product between real-space and reciprocal-space vectors for IDFT
        r_dot_k = np.einsum('ri,ki->kr', self.r_vecs, self.k_vecs).reshape((1,)*axis + (self.nk,)*2 + (1,)*(A.ndim-1-axis))
        
        # Perform the IDFT using the inverse of the DFT formula
        return np.sum(np.exp(-1j*r_dot_k) * A.reshape(A.shape[:axis+1] + (1,) + A.shape[axis+1:]), axis) / np.sqrt(self.nk)