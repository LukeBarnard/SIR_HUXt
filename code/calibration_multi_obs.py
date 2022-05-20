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
import sir_huxt_multi_obs as sir
import SIR_HUXt_plots as sirplt


def calibrate_multi_obs():
    """
    Run the SIR scheme repeatedly for guesses at one truth CME and different realisations of noise added to the observations.
    Use a uniform solar wind background.
    """ 
    # Test the SIR scheme
    np.random.seed(2584802)
    
    model = sir.setup_uniform_huxt(dt_scale=20)
    
    # Generate a "truth" CME
    base_cme = sir.get_base_cme()
    
    # Get HUXt solution of this truth CME, and observations from L5
    model.solve([base_cme])
    cme_truth = model.cmes[0]
    
    obs1_lon = -60*u.deg
    L5Obs = sir.Observer(model, cme_truth, obs1_lon, el_min=4.0, el_max=30.0)

    obs2_lon = 1*u.deg
    L1Obs = sir.Observer(model, cme_truth, obs2_lon, el_min=4.0, el_max=30.0)

    fig, ax = sirplt.plot_huxt_with_observer(model.time_out[8], model, [L5Obs, L1Obs], add_flank=True, add_fov=True)
    fig.savefig('multi_obs_truth.png')
    plt.close('all')

    # Make directory to store this experiment in
    dirs = sir.get_project_dirs()
    output_dir = 'calibrate_sir_multi_obs'
    output_dir = os.path.join(dirs['sir_analysis'], output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # Run the SIR scheme on this event many times to see how the performance is
    n_ens = 50
    n_runs = 100
    for i in range(n_runs):
    
        # Make a guess at the CME initial values 
        cme_guess = sir.perturb_cme(base_cme)
        
        # Low observational error
        observed_cme_flank = L5Obs.compute_synthetic_obs(el_spread=0.1, cadence=1, el_min=4.0, el_max=30.0)
        obs1 = {'observer_lon':obs1_lon, 'observed_cme_flank':observed_cme_flank, 'truth_cme_params':cme_truth.parameter_array()}

        observed_cme_flank = L1Obs.compute_synthetic_obs(el_spread=0.1, cadence=1, el_min=4.0, el_max=30.0)
        obs2 = {'observer_lon':obs2_lon, 'observed_cme_flank':observed_cme_flank, 'truth_cme_params':cme_truth.parameter_array()}

        tag = "sir_multi_run_{:03d}".format(i)
        sir.SIR(model, cme_guess, obs1, obs2, n_ens, output_dir, tag)
      
    return

    
if __name__ == "__main__":
    
    calibrate_multi_obs()
   
    
    
    