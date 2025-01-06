from . import minimisation as mn
from . import mcmc as mc
from . import spectro as sp
import numpy as np

def main_mcmc():
    s = [0.0001, 10, 0.001, 5, 2]
    sigma = np.array(s)
    mcmc = mc.MCMC(sigma)
    mcmc.Main_mcmc()
    
def main_multi_mcmc():
    s = [0.0001, 10, 0.001, 5, 2]
    sigma = np.array(s)
    mcmc = mc.MCMC(sigma, True)
    mcmc.Main_mcmc()
    
def main_emcee():
    s = [0.0001, 10, 0.001, 5, 2]
    sigma = np.array(s)
    mcmc = mc.MCMC(sigma)
    mcmc.Main_emcee()

def main_min():
    min = mn.Minimisation()
    min.Main()
    
def main_sp():
    spectre = sp.SpectroAnalysis()
    spectre.run_analysis()
