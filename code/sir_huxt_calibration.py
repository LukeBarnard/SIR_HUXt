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


def run_experiment_uniform_wind():
    
    # Test the SIR scheme
    np.random.seed(20100114)

    start_time = Time('2008-06-30T00:00:00')

    for i in range(100):

        model = sir.setup_huxt(start_time, uniform_wind=True)

        # Generate a "truth" CME
        base_cme = sir.get_base_cme()
        cme_truth = sir.perturb_cme(base_cme)

        # Get HUXt solution of this truth CME, and observations from L5
        model.solve([cme_truth])
        cme_truth = model.cmes[0]
        hit, t_arrive, t_transit, hit_lon, hit_id = cme_truth.compute_arrival_at_body('EARTH')

        observer_lon = -60*u.deg
        L5Obs = sir.Observer(model, cme_truth, observer_lon, el_min=4.0, el_max=30.0)

        model_flank = L5Obs.model_flank
        observed_cme_flank = L5Obs.compute_synthetic_obs(el_spread=0.5, cadence=3, el_min=4.0, el_max=30.0)

        observations = {'t_arrive':t_arrive, 't_transit':t_transit, 'observer_lon':observer_lon,
                        'observed_cme_flank':observed_cme_flank, 'cme_params':cme_truth.parameter_array()}

        tag = "uniform_weak_run_{:03d}".format(i)
        sir.SIR(model, base_cme, observations, tag)
        
        return
    
    
if __name__ == "__main__":
    
    run_experiment_uniform_wind()