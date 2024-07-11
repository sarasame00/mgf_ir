import numpy as np
import sparse_ir as ir
import mgf_ir



# Parameters
xiSO = 0.02
gJT = 0.5
theta = 0
pdd = 0.13
ddd = 0.05
pol = 'L'
racahA = 6.4
racahB = 0.12
racahC = 0.552
beta = 45
wM = 4
nk_lin = 8





def angular_momentum_t2g():
    a = np.arange(3)
    y,x,z = np.meshgrid(a,a,a)
    
    fwd = ((x==0) * (y==1) * (z==2) +
           (x==1) * (y==2) * (z==0) +
           (x==2) * (y==0) * (z==1))
    bkd = ((x==0) * (y==2) * (z==1) +
           (x==1) * (y==0) * (z==2) +
           (x==2) * (y==1) * (z==0))
    
    return 1j * fwd - 1j * bkd

lx,ly,lz = angular_momentum_t2g()


sx,sy,sz = np.array([[0.,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.eye(2)

HSOC = 0.5*xiSO * (np.kron(lx,sx) + np.kron(ly,sy) + np.kron(lz,sz))



HJT = np.kron(-gJT/np.sqrt(3) * (lx @ lx - ly @ ly) * np.sin(theta) - gJT * (lz @ lz - 2/3*np.eye(3)) * np.cos(theta), np.eye(2))


Hloc = HSOC + HJT


def moment_tensor(C,D):
    # t[component,hop direction,d-orbital,p-orbital]
    # d-basis {z,h,t,u,v}
    k = np.sqrt(3)
    t = np.zeros((3,3,3,3), dtype=np.complex128)

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

    return -1j*t


P = moment_tensor(0.13, 0.05)
vec_pot = np.array([1,0,0]) # {'L':np.array([1,1j,0]), 'R':np.array([1,-1j,0])}[pol]
Hhop = np.kron(np.einsum('l,lkap,lkpb->kab', vec_pot, P, P.conjugate()), np.eye(2))



def coulomb_interaction_d_Oh(A, B, C):
    F_4 = C/35
    F_2 = B + 5*F_4
    F_0 = A + 49*F_4
    
    F0 = F_0
    F2 = F_2/49
    F4 = F_4/441
    
    #Orbital basis
    sq2 = np.sqrt(2)
    orb_basis = np.array([[0,0,-1j/sq2,0,1/sq2],
                          [1j/sq2,-1/sq2,0,0,0],
                          [0,0,0,1,0],
                          [1j/sq2,1/sq2,0,0,0],
                          [0,0,1j/sq2,0,1/sq2]], dtype=np.complex128) # Transformation matrix
    
    #Slater integrals
    c2 = np.array([[-2,np.sqrt(6),-2,0,0],
                    [-np.sqrt(6),1,1,-np.sqrt(6),0],
                    [-2,-1,2,-1,-2],
                    [0,-np.sqrt(6),1,1,-np.sqrt(6)],
                    [0,0,-2,np.sqrt(6),-2]]) / 7 

    c4 = np.array([[1,-np.sqrt(5),np.sqrt(15),-np.sqrt(35),np.sqrt(70)],
                    [np.sqrt(5),-4,np.sqrt(30),-np.sqrt(40),np.sqrt(35)],
                    [np.sqrt(15),-np.sqrt(30),6,-np.sqrt(30),np.sqrt(15)],
                    [np.sqrt(35),-np.sqrt(40),np.sqrt(30),-4,np.sqrt(5)],
                    [np.sqrt(70),-np.sqrt(35),np.sqrt(15),-np.sqrt(5),1]]) / 21
    
    Vc = np.zeros((6,6,6,6), dtype=np.complex128)
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

                    if i==j or k==l: # Pauli exclusion principle
                        continue

                    else:
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
                                    # print(i,j,k,l, oi,oj,ok,ol, m1, m2, m3)
                                    # print(vi[2-m1])
                                    # print(vj[2-m2])
                                    # print(vk[2-m3])
                                    # print(vl[2-m4])
                                    # print(c2[2-m1,2-m3])
                                    # print(c2[2-m2,2-m4])
                                    # print(c4[2-m1,2-m3])
                                    # print(c4[2-m2,2-m4])
                                    assert not np.any(np.isnan(Vc))
    return Vc.real


V = coulomb_interaction_d_Oh(racahA, racahB, racahC)



H = mgf_ir.Hamiltonian(Hloc, V, Hhop)



irb = ir.FiniteTempBasis('F', beta, wM)
dy_solver = mgf_ir.DysonSolver(H, irb, 4, negf_ir.Lattice(*(nk_lin,)*3))
dy_solver.solve()
