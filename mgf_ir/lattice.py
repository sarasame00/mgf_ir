import numpy as np





class Lattice:
    def __init__(self, nk1, nk2, nk3, basis_kind="cubic", a=None, b=None, c=None):
        self.__nk1 = nk1
        self.__nk2 = nk2
        self.__nk3 = nk3
        
        self.__r_basis, self.__k_basis = self.__basis_set(basis_kind, a, b, c)
        
        self.__r_vecs = np.einsum('ij,...j->...i', self.__r_basis, np.swapaxes(np.array(np.meshgrid(np.arange(nk1), np.arange(nk2), np.arange(nk3))), 1,2).reshape(3,nk1*nk2*nk3).T)
        
        self.__k_vecs = np.einsum('ij,...j->...i', self.__k_basis, np.swapaxes(np.array(np.meshgrid(np.arange(nk1), np.arange(nk2), np.arange(nk3))), 1,2).reshape(3,nk1*nk2*nk3).T/np.array([nk1,nk2,nk3]).reshape((1,3)))
        # self.__k_vecs = np.swapaxes(np.swapaxes(np.array(np.meshgrid(np.arange(nk1),np.arange(nk2),np.arange(nk3))), 1, 2).reshape((3,nk1*nk2*nk3)), 0, 1) * 2*np.pi/np.array([nk1,nk2,nk3]).reshape((1,3))
    
    @property
    def nk(self):
        return self.__nk1 * self.__nk2 * self.__nk3
    
    @property
    def a1(self):
        return self.__r_basis[:,0]
    
    @property
    def a2(self):
        return self.__r_basis[:,1]
    
    @property
    def a3(self):
        return self.__r_basis[:,2]
    
    @property
    def b1(self):
        return self.__k_basis[:,0]
    
    @property
    def b2(self):
        return self.__k_basis[:,1]
    
    @property
    def b3(self):
        return self.__k_basis[:,2]
    
    @property
    def r_vecs(self):
        return self.__r_vecs
    
    @property
    def rij(self):
        # Returns rij = Rj - Ri
        return self.r_vecs[None,:,:] - self.r_vecs[:,None,:]
    
    @property
    def k_vecs(self):
        return self.__k_vecs
    
    def __basis_set(self, kind, a, b, c):
        if kind=="cubic":
            r_basis = np.eye(3)
        else:
            raise NotImplementedError()
        
        k_basis = np.zeros_like(r_basis)
        vol = np.dot(np.cross(r_basis[:,0],r_basis[:,1]),r_basis[:,2])
        for i in range(3):
            j = (i+1)%3
            k = (i-1)%3
            k_basis[:,i] = 2*np.pi * np.cross(r_basis[:,j],r_basis[:,k]) / vol
        
        return r_basis, k_basis
    
    def dft(self, A, axis=-1):
        if A.shape[axis] != self.nk:
            raise ValueError("Axis of transformation must have the same size as lattice")
        if axis < 0:
            axis += A.ndim
        r_dot_k = np.einsum('ri,ki->rk', self.r_vecs, self.k_vecs).reshape((1,)*axis + (self.nk,)*2 + (1,)*(A.ndim-1-axis))
        return np.sum(np.exp(1j*r_dot_k) * A.reshape(A.shape[:axis+1] + (1,) + A.shape[axis+1:]), axis) / np.sqrt(self.nk)
    
    def idft(self, A, axis=-1):
        if A.shape[axis] != self.nk:
            raise ValueError("Axis of transformation must have the same size as lattice")
        if axis < 0:
            axis += A.ndim
        r_dot_k = np.einsum('ri,ki->kr', self.r_vecs, self.k_vecs).reshape((1,)*axis + (self.nk,)*2 + (1,)*(A.ndim-1-axis))
        return np.sum(np.exp(-1j*r_dot_k) * A.reshape(A.shape[:axis+1] + (1,) + A.shape[axis+1:]), axis) / np.sqrt(self.nk)