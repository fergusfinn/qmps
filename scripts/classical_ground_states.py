"""classica"""
import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('pub_fast')

def example_DMRG_heisenberg_xxz_infinite(Jx=1, Jy=1, Jz=1, hx=0, hy=0, hz=0, conserve='best', verbose=True, chi_max=100, S=0.5):
    print("infinite DMRG, Heisenberg XXZ chain")
    print("Jz={Jz:.2f}, conserve={conserve!r}".format(Jz=Jz, conserve=conserve))
    model_params = dict(
        L=2,
        S=S,  # spin 1/2
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,  # couplings
        hx=hx,
        hy=hy,
        hz=hz,
        bc_MPS='infinite',
        conserve=conserve,
        verbose=verbose)
    M = SpinModel(model_params)
    product_state = ["up", "down"]  # initial Neel state
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1.e-10,
        },
        'max_E_err': 1.e-10,
        'verbose': verbose,
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
    print("E = {E:.13f}".format(E=E))
    print("final bond dimensions: ", psi.chi)
    Sz = psi.expectation_value("Sz")  # Sz instead of Sigma z: spin-1/2 operators!
    mag_z = np.mean(Sz)
    print("<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(Sz0=Sz[0],
                                                                     Sz1=Sz[1],
                                                                     mag_z=mag_z))
    # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
    print("correlation length:", psi.correlation_length())
    corrs = psi.correlation_function("Sz", "Sz", sites1=range(10))
    print("correlations <Sz_i Sz_j> =")
    print(corrs)
    return E, psi, M

def 
if __name__=='__main__':
    chi = 2
    e2, _, _ = example_DMRG_heisenberg_xxz_infinite(Jx=0, Jy=0, Jz=4., hx=2., hy=0, hz=0, chi_max = chi, S=0.5)
    print(e2)
    #chis = (2**np.arange(1, 8)).astype(int)
    #print(chis)
    #e1s = []
    #e2s = []
    #e3s = []
    #e4s = []
    #for chi in chis:
    #    e3, _, _ = example_DMRG_heisenberg_xxz_infinite(Jx=-4, Jy=-4, Jz=0.01, hx=0., hy=0, hz=0, chi_max = chi, S=0.5)
    #    e1, _, _ = example_DMRG_heisenberg_xxz_infinite(Jx=-4, Jy=-4, Jz=-4., hx=0., hy=0, hz=0, chi_max = chi, S=0.5)
    #    e2, _, _ = example_DMRG_heisenberg_xxz_infinite(Jx=0, Jy=0, Jz=-4., hx=2., hy=0, hz=0, chi_max = chi, S=0.5)
    #    e4, _, _ = example_DMRG_heisenberg_xxz_infinite(Jx=0.5, Jy=0, Jz=-1., hx=1., hy=1, hz=0, chi_max = chi, S=0.5)
    #    e1s.append(e1)
    #    e2s.append(e2)
    #    e3s.append(e3)
    #    e4s.append(e4)
    #    print(e1, e2, e3, e4)
    #e1s = np.array(e1s)
    #e2s = np.array(e2s)
    #e3s = np.array(e3s)
    #e4s = np.array(e4s)
    #plt.plot(chis[:-1], (e1s-e1s[-1])[:-1], label='xxz')
    #plt.plot(chis[:-1], (e2s-e2s[-1])[:-1], label='tfi')
    #plt.plot(chis[:-1], (e3s-e3s[-1])[:-1], label='xy')
    #plt.plot(chis[:-1], (e4s-e4s[-1])[:-1], label='nc')
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylabel('$E-E_0$')
    #plt.xlabel('D')
    #plt.legend()
    #plt.show()
    #plt.savefig('../images/noise/convergence.pdf')

