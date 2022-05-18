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
import sir_huxt_lon as shl
import sir_huxt_v as shv
import sir_huxt_width as shw


def calibrate_shv():
    """
    Run the SIR scheme repeatedly for guesses at one truth CME and different realisations of noise added to the observations.
    Use a uniform solar wind background.
    """ 
    # Test the SIR scheme
    np.random.seed(190802)
    
    model = shv.setup_uniform_huxt(dt_scale=20)
    
    # Generate a "truth" CME
    base_cme = shv.get_base_cme()
    
    # Get HUXt solution of this truth CME, and observations from L5
    model.solve([base_cme])
    cme_truth = model.cmes[0]
    
    observer_lon = -60*u.deg
    L5Obs = shv.Observer(model, cme_truth, observer_lon, el_min=4.0, el_max=35.0)
    
    # Make directory to store this experiment in
    dirs = shv.get_project_dirs()
    output_dir = 'shv_calibrate_v2'
    output_dir = os.path.join(dirs['sir_analysis'], output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # Run the SIR scheme on this event many times to see how the performance is
    n_ens = 20
    n_runs = 50
    for i in range(n_runs):
    
        # Make a guess at the CME initial values 
        cme_guess = shv.perturb_cme(base_cme)
        
        # Low observational error
        observed_cme_flank = L5Obs.compute_synthetic_obs(el_spread=0.1, cadence=1, el_min=4.0, el_max=35.0)
    
        observations = {'observer_lon':observer_lon, 'observed_cme_flank':observed_cme_flank, 'truth_cme_params':cme_truth.parameter_array()}

        tag = "shv_run_{:03d}".format(i)
        shv.SIR(model, cme_guess, observations, n_ens, output_dir, tag)
      
    return


def calibrate_shl():
    """
    Run the SIR scheme repeatedly for guesses at one truth CME and different realisations of noise added to the observations.
    Use a uniform solar wind background.
    """ 
    # Test the SIR scheme
    np.random.seed(196308)
    
    start_time = Time('2008-06-30T00:00:00')
    
    model = shl.setup_uniform_huxt(dt_scale=20)
    
    # Generate a "truth" CME
    base_cme = shl.get_base_cme()
    
    # Get HUXt solution of this truth CME, and observations from L5
    model.solve([base_cme])
    cme_truth = model.cmes[0]
    
    observer_lon = -60*u.deg
    L5Obs = shl.Observer(model, cme_truth, observer_lon, el_min=4.0, el_max=35.0)
    
    # Make directory to store this experiment in
    dirs = shl.get_project_dirs()
    output_dir = 'shl_calibrate_v2'
    output_dir = os.path.join(dirs['sir_analysis'], output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # Run the SIR scheme on this event many times to see how the performance is
    n_ens = 20
    n_runs = 50
    for i in range(n_runs):
    
        # Make a guess at the CME initial values 
        cme_guess = shl.perturb_cme(base_cme)
        
        # Low observational error
        observed_cme_flank = L5Obs.compute_synthetic_obs(el_spread=0.1, cadence=1, el_min=4.0, el_max=35.0)
    
        observations = {'observer_lon':observer_lon, 'observed_cme_flank':observed_cme_flank, 'truth_cme_params':cme_truth.parameter_array()}

        tag = "shl_run_{:03d}".format(i)
        shl.SIR(model, cme_guess, observations, n_ens, output_dir, tag)
      
    return


def calibrate_shw():
    """
    Run the SIR scheme repeatedly for guesses at one truth CME and different realisations of noise added to the observations.
    Use a uniform solar wind background.
    """ 
    # Test the SIR scheme
    np.random.seed(19630802)
    
    start_time = Time('2008-06-30T00:00:00')
    
    model = shw.setup_uniform_huxt(dt_scale=20)
    
    # Generate a "truth" CME
    base_cme = shw.get_base_cme()
    
    # Get HUXt solution of this truth CME, and observations from L5
    model.solve([base_cme])
    cme_truth = model.cmes[0]
    
    observer_lon = -60*u.deg
    L5Obs = shw.Observer(model, cme_truth, observer_lon, el_min=4.0, el_max=35.0)
    
    # Make directory to store this experiment in
    dirs = shw.get_project_dirs()
    output_dir = 'shw_calibrate_v2'
    output_dir = os.path.join(dirs['sir_analysis'], output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # Run the SIR scheme on this event many times to see how the performance is
    n_ens = 20
    n_runs = 50
    for i in range(n_runs):
    
        # Make a guess at the CME initial values 
        cme_guess = shw.perturb_cme(base_cme)
        
        # Low observational error
        observed_cme_flank = L5Obs.compute_synthetic_obs(el_spread=0.1, cadence=1, el_min=4.0, el_max=35.0)
    
        observations = {'observer_lon':observer_lon, 'observed_cme_flank':observed_cme_flank, 'truth_cme_params':cme_truth.parameter_array()}

        tag = "shw_run_{:03d}".format(i)
        shw.SIR(model, cme_guess, observations, n_ens, output_dir, tag)
      
    return

    
if __name__ == "__main__":
    
    calibrate_shv()
    calibrate_shl()
    calibrate_shw()
    
    
    