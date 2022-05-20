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
import sir_huxt_mono_obs as sir

def calibrate_structured_wind():
    """
    Run the SIR scheme repeatedly for guesses at one truth CME and different realisations of noise added to the observations.
    Use a uniform solar wind background.
    """ 
    # Test the SIR scheme
    np.random.seed(19084802)
    
    start_time = Time('2008-01-01T00:00:00')
    stop_time = Time('2010-01-01T00:00:00')
    start_days = np.linspace(np.fix(start_time.jd), np.fix(stop_time.jd), 100)

    for i in range(days.size):
        start_time = Time(days[i], format='jd')
        
        dt_scale = 20
        model = sir.setup_huxt(start_time, dt_scale, uniform_wind=False)

        # Generate a "truth" CME
        base_cme = sir.get_base_cme()

        # Get HUXt solution of this truth CME, and observations from L5
        model.solve([base_cme])
        cme_truth = model.cmes[0]

        observer_lon = -60*u.deg
        L5Obs = sir.Observer(model, cme_truth, observer_lon, el_min=4.0, el_max=35.0)

        # Make directory to store this experiment in
        dirs = sir.get_project_dirs()
        output_dir = 'structured_wind'
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
            observed_cme_flank = L5Obs.compute_synthetic_obs(el_spread=0.1, cadence=1, el_min=4.0, el_max=35.0)

            observations = {'observer_lon':observer_lon, 'observed_cme_flank':observed_cme_flank, 
                            'truth_cme_params':cme_truth.parameter_array()}

            tag = "structured_wind_{:03d}".format(i)
            sir.SIR(model, cme_guess, observations, n_ens, output_dir, tag)
      
    return

    
if __name__ == "__main__":
    
    calibrate_structured_wind()
   
    
    
    