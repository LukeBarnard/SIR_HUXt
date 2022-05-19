import os
import glob

from astropy.time import Time
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.stats as st
from sklearn.neighbors import KernelDensity

import huxt as H
import huxt_analysis as HA
import sir_huxt_v_width as shvw

def calibrate_shvw():
    """
    Run the SIR scheme repeatedly for guesses at one truth CME and different realisations of noise added to the observations.
    Use a uniform solar wind background.
    """ 
    # Test the SIR scheme
    np.random.seed(190802)
    
    model = shvw.setup_uniform_huxt(dt_scale=20)
    
    # Generate a "truth" CME
    base_cme = shvw.get_base_cme()
    
    # Get HUXt solution of this truth CME, and observations from L5
    model.solve([base_cme])
    cme_truth = model.cmes[0]
    
    observer_lon = -60*u.deg
    L5Obs = shvw.Observer(model, cme_truth, observer_lon, el_min=4.0, el_max=35.0)
    
    # Make directory to store this experiment in
    dirs = shvw.get_project_dirs()
    output_dir = 'shvw_calibrate'
    output_dir = os.path.join(dirs['sir_analysis'], output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # Run the SIR scheme on this event many times to see how the performance is
    n_ens = 30
    n_runs = 100
    for i in range(n_runs):
    
        # Make a guess at the CME initial values 
        cme_guess = shvw.perturb_cme(base_cme)
        
        # Low observational error
        observed_cme_flank = L5Obs.compute_synthetic_obs(el_spread=0.1, cadence=1, el_min=4.0, el_max=35.0)
    
        observations = {'observer_lon':observer_lon, 'observed_cme_flank':observed_cme_flank, 'truth_cme_params':cme_truth.parameter_array()}

        tag = "shvw_run_{:03d}".format(i)
        shvw.SIR(model, cme_guess, observations, n_ens, output_dir, tag)
      
    return

    
if __name__ == "__main__":
    
    calibrate_shvw()
   
    
    
    