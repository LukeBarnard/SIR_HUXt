import huxt as H
import huxt_analysis as HA
import SIR_HUXt as sir

import astropy.units as u
import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.stats as st

from sklearn.neighbors import KernelDensity
from astropy.time import Time


def experiment_uniform_wind_low_obs_error():
    """
    Run the SIR scheme repeatedly for guesses at one truth CME and different realisations of noise added to the observations.
    Use a uniform solar wind background.
    """ 
    # Test the SIR scheme
    np.random.seed(19630802)
    
    start_time = Time('2008-06-30T00:00:00')
    
    model = sir.setup_huxt(start_time, uniform_wind=True)
    
    # Generate a "truth" CME
    base_cme = sir.get_base_cme()
    
    # Get HUXt solution of this truth CME, and observations from L5
    model.solve([base_cme])
    cme_truth = model.cmes[0]
    hit, t_arrive, t_transit, hit_lon, hit_id = cme_truth.compute_arrival_at_body('EARTH')
    
    observer_lon = -60*u.deg
    L5Obs = sir.Observer(model, cme_truth, observer_lon, el_min=4.0, el_max=30.0)
    
    # Run the SIR scheme on this event many times to see how the performance is
    n_ens = 20
    n_runs = 100
    for i in range(n_runs):
    
        # Make a guess at the CME initial values 
        cme_guess = sir.perturb_cme(base_cme)
        
        # Low observational error
        observed_cme_flank = L5Obs.compute_synthetic_obs(el_spread=0.1, cadence=3, el_min=4.0, el_max=30.0)
    
        observations = {'t_arrive':t_arrive, 't_transit':t_transit, 'observer_lon':observer_lon,
                        'observed_cme_flank':observed_cme_flank, 'cme_params':cme_truth.parameter_array()}
    
        tag = "uniform_low_error_n{:03d}_lowerllhd_lowerrsmp_run_{:03d}".format(n_ens, i)
        sir.SIR(model, cme_guess, observations, n_ens, tag)
      
        
    return
    

def experiment_real_wind_low_obs_error():
    """
    Run the SIR scheme repeatedly for guesses at one truth CME and different realisations of noise added to the observations.
    Include realistic background solar wind structure
    """

    # Test the SIR scheme
    np.random.seed(19851013)
    
    start_time = Time('2008-06-01T00:00:00')
    
    model = sir.setup_huxt(start_time, uniform_wind=False)
    
    # Generate a "truth" CME
    base_cme = sir.get_base_cme()
    cme_truth = sir.perturb_cme(base_cme)
    
    # Get HUXt solution of this truth CME, and observations from L5
    model.solve([cme_truth])
    cme_truth = model.cmes[0]
    hit, t_arrive, t_transit, hit_lon, hit_id = cme_truth.compute_arrival_at_body('EARTH')
    
    observer_lon = -60*u.deg
    L5Obs = sir.Observer(model, cme_truth, observer_lon, el_min=4.0, el_max=30.0)
    
    # Run the SIR scheme on this event many times to see how the performance is
    n_ens = 20
    n_runs = 100
    for i in range(n_runs):
    
        # Make a guess at the CME initial values 
        cme_guess = sir.perturb_cme(cme_truth)
        
        # Low observational error
        observed_cme_flank = L5Obs.compute_synthetic_obs(el_spread=0.1, cadence=3, el_min=4.0, el_max=30.0)
    
        observations = {'t_arrive':t_arrive, 't_transit':t_transit, 'observer_lon':observer_lon,
                        'observed_cme_flank':observed_cme_flank, 'cme_params':cme_truth.parameter_array()}
    
        tag = "real_low_error_n{:03d}_run_{:03d}".format(n_ens, i)
        sir.SIR(model, cme_guess, observations, n_ens, tag)
        
        
    return

    
if __name__ == "__main__":
    
    experiment_uniform_wind_low_obs_error()
    #experiment_real_wind_low_obs_error()
    
    