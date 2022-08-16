import os
import glob
import itertools
import multiprocessing

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

def calibration_osse(params):
    """
    Run the SIR scheme repeatedly for guesses at one truth CME and different realisations of noise added to the observations.
    Use a uniform solar wind background.
    """ 
    scenario = params[0]
    observer_lon = params[1]
    
    print("Running {} scenario at {} observer longitude".format(scenario, observer_lon))
    
    # Set seed so raw ensemble is the same for all OSSEs
    np.random.seed(19079502)
    
    model, model1d = sir.setup_huxt(dt_scale=20)
    
    # Generate a "truth" CME
    base_cme = sir.get_cme_scenario(model, scenario)
    
    # Get HUXt solution of this truth CME, and observations
    model.solve([base_cme])
    cme_truth = model.cmes[0]
    
    # Also do hi-res 1d run for arrival time at Earth
    model1d.solve([base_cme])
    cme_arr = model1d.cmes[0]
    arrival_stats = cme_arr.compute_arrival_at_body('EARTH')
    hit = arrival_stats['hit']
    if arrival_stats['hit']:
        t_transit = arrival_stats['t_transit'].to(u.s).value
        v_hit = arrival_stats['v'].value
    else:
        t_transit = np.NaN
        v_hit = np.NaN
    
    Obs = sir.Observer(model, cme_truth, observer_lon*u.deg, el_min=4.0, el_max=35.0)
    
    # Make directory to store this experiment in
    dirs = sir.get_project_dirs()
    if observer_lon < 0:
        lon_out = observer_lon + 360
    else:
        lon_out = observer_lon
        
    output_dir = 'obs_lon_{:03d}_cme_{}_mp'.format(lon_out, scenario)
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
        observed_cme_flank = Obs.compute_synthetic_obs(el_spread=0.1, cadence=1, el_min=4.0, el_max=35.0)
    
        observations = {'observer_lon':observer_lon*u.deg, 'observed_cme_flank':observed_cme_flank, 'truth_cme_params':cme_truth.parameter_array(), 't_transit':t_transit, 'v_hit':v_hit}

        tag = "run_{:03d}".format(i)
        sir.SIR(model, model1d, cme_guess, observations, n_ens, output_dir, tag)
      
    return

    
if __name__ == "__main__":
    
    lons = [-90, -80, -70, -60, -50, -40, -30, -20]
    scenarios = sir.load_cme_scenarios()
    params = itertools.product(scenarios.keys(), lons)
    paramlist = list(params)
    
    #Generate processes equal to the number of cores
    pool = multiprocessing.Pool()
    res  = pool.map(calibration_osse, paramlist)
            