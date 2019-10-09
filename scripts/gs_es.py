import numpy as np

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('pub_fast')

def example_DMRG_heisenberg_xxz_infinite(Jx=1, Jy=1, Jz=1, hx=0, hy=0, hz=0, conserve='best', verbose=True, chi_max=100, S=0.5):
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
    Sz = psi.expectation_value("Sz")  # Sz instead of Sigma z: spin-1/2 operators!
    mag_z = np.mean(Sz)
    # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
    return E, psi, M


if __name__=='__main__':
    N = 50
    chi = 2
    _, _, _ = example_DMRG_heisenberg_xxz_infinite(Jx=0, Jy=0, Jz=-4., hx=2., hy=0, hz=0, chi_max = chi, S=0.5)

