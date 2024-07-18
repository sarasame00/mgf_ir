import numpy as np
import sparse_ir as ir


def mats_ir(H, Hkin, V, nk_lin, mu, beta, wM, particle):
    if particle=='B':
        ps = 1
    elif particle=='F':
        ps = -1
    else:
        raise ValueError("Particle only 'B' (boson) or 'F' (fremion)")
    
    nk = nk_lin**3
    
    eigH, basisH = np.linalg.eigh(H)
    basisHh = basisH.T.conjugate()
    hoppings = (basisHh @ Hkin @ basisH).real
    Umf = -np.einsum('bk,am,na,ld,mkln', basisHh, basisHh, basisH, basisH, V - np.swapaxes(V, 2, 3)).real
    
    k_vecs = np.swapaxes(np.swapaxes(np.array(np.meshgrid(*(np.arange(nk_lin),)*3)), 1, 2).reshape((3,nk)), 0, 1) * 2*np.pi/nk_lin
    
    irb = ir.FiniteTempBasis('F', beta, wM, 1e-8)
    siw = ir.MatsubaraSampling(irb)
    iw = 1j*np.pi/beta * siw.wn
    stau = ir.TauSampling(irb)
    u_beta = irb.u(beta)
    
    # [w,k,a]
    hk = eigH[None,:] + np.sum(hoppings[:,None,:] * np.cos(k_vecs.T[:,:,None]), axis=0) - mu
    giwk = 1/(iw[:,None,None] - hk[None,:,:])
    glk = siw.fit(giwk)
    Glk = np.copy(glk)
    while True:
        sehf = ps*np.einsum('ab,lkb,l->a', Umf, Glk, u_beta) / nk
        Giwk = siw.fit(Glk)
        Giwk = giwk + np.einsum('a,wak,wak', sehf, giwk, Giwk)
        Glk = siw.evaluate(Giwk)
        # Add condition to stop loop
    
    Gloc_l = np.sum(Glk, axis=1) / nk # Sum over k
    N0 = -np.sum(Gloc_l, u_beta[:,None])