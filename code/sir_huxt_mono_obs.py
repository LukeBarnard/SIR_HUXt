import errno
import glob
import os

from astropy.time import Time
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy.coordinates.sun as sn
import scipy.stats as st
from sklearn.neighbors import KernelDensity


import huxt as H
import huxt_inputs as Hin
import huxt_analysis as Ha
import sir_huxt_plots as sirplt


class Observer:
    """
    The function of this observer class is to provide pseudo-observations of a ConeCME's flank elongation from a HUXt simulation
    for an observer at a specified longitude relative to Earth. The observers distance defaults to 1 AU, and has the same latitude as Earth
    during the HUXt run. 
    """
    
    @u.quantity_input(longitude=u.deg)
    def __init__(self, model, cme, longitude, el_min=4.0, el_max=30.0):
        
        ert_ephem = model.get_observer('EARTH')
        
        self.time = ert_ephem.time 
        self.r = ert_ephem.r*0 + 1*u.AU
        self.lon = ert_ephem.lon + longitude
        self.lat = ert_ephem.lat
        self.el_min = el_min
        self.el_max = el_max
        # Force longitude into 0-360 domain
        id_over = self.lon > 360*u.deg
        id_under = self.lon < 0*u.deg
        if np.any(id_over):
            self.lon[id_over] = self.lon[id_over] - 360*u.deg
        if np.any(id_under):
            self.lon[id_under] = self.lon[id_under] + 360*u.deg
        
        self.model_flank = self.compute_flank_profile(cme)
        
        
    def compute_flank_profile(self, cme):
        """
        Compute the time elongation profile of the flank of a ConeCME in HUXt. The observer longtidue is specified
        relative to Earth but otherwise matches Earth's coords.

        Parameters
        ----------
        cme: A ConeCME object from a completed HUXt run (i.e the ConeCME.coords dictionary has been populated).
        Returns
        -------
        obs_profile: Pandas dataframe giving the coordinates of the ConeCME flank from STA's perspective, including the
                    time, elongation, position angle, and HEEQ radius and longitude.
        """
        times = Time([coord['time'] for i, coord in cme.coords.items()])

        # Compute observers location using earth ephem, adding on observers longitude offset from Earth
        # and correct for runover 2*pi
        flank = pd.DataFrame(index=np.arange(times.size), columns=['time', 'el', 'r', 'lon'])
        flank['time'] = times.jd

        for i, coord in cme.coords.items():

            if len(coord['r']) == 0:
                flank.loc[i, ['lon', 'r', 'el']] = np.NaN
                continue

            r_obs = self.r[i]
            x_obs = self.r[i] * np.cos(self.lat[i]) * np.cos(self.lon[i])
            y_obs = self.r[i] * np.cos(self.lat[i]) * np.sin(self.lon[i])
            z_obs = self.r[i] * np.sin(self.lat[i])

            lon_cme = coord['lon']
            lat_cme = coord['lat']
            r_cme = coord['r']

            x_cme = r_cme * np.cos(lat_cme) * np.cos(lon_cme)
            y_cme = r_cme * np.cos(lat_cme) * np.sin(lon_cme)
            z_cme = r_cme * np.sin(lat_cme)
            #############
            # Compute the observer CME distance, S, and elongation

            x_cme_s = x_cme - x_obs
            y_cme_s = y_cme - y_obs
            z_cme_s = z_cme - z_obs
            s = np.sqrt(x_cme_s**2 + y_cme_s**2 + z_cme_s**2)

            numer = (r_obs**2 + s**2 - r_cme**2).value
            denom = (2.0 * r_obs * s).value
            e_obs = np.arccos(numer / denom)
            
            # Restrict those CME points to those in FOV
            # For those ahead of Earth, this is negative y_cme_s
            # For those behind Earth, this is positive y_cme_s
            if self.lon[i] < np.pi*u.rad:
                id_sub = y_cme_s.value < 0
                e_obs = e_obs[id_sub]
                lon_cme = lon_cme[id_sub]
                r_cme = r_cme[id_sub]
            elif self.lon[i] > np.pi*u.rad:
                id_sub = y_cme_s.value > 0
                e_obs = e_obs[id_sub]
                lon_cme = lon_cme[id_sub]
                r_cme = r_cme[id_sub]

            # Find the flank coordinate and update output
            id_obs_flank = np.argmax(e_obs)       
            flank.loc[i, 'lon'] = lon_cme[id_obs_flank].value
            flank.loc[i, 'r'] = r_cme[id_obs_flank].value
            flank.loc[i, 'el'] = np.rad2deg(e_obs[id_obs_flank])

        # Force values to be floats.
        keys = ['lon', 'r', 'el']
        flank[keys] = flank[keys].astype(np.float64)
        return flank
    
    
    def compute_synthetic_obs(self, el_spread=0.1, cadence=1, el_min=4.0, el_max=30.0):
        """
        Return synthetic observations with a specified Gaussian uncertainty spread, cadence, and maximum elongation.
        el_spread = standard deviation of random Gaussian noise added to the modelled elongation.
        cadence = The cadence with witch observations are returned, as a whole number of model time steps.
        el_min = The minimum elongation of the observers field of view.
        el_max = The maximum elongation of the observers field of view.
        """

        # Compute the time-elongation profiles of the CME flanks from STA and STB
        model_flank = self.model_flank.copy()

        # Remove invalid points
        model_flank.dropna(inplace=True)

        # Add observation noise.
        obs_flank = model_flank.loc[:, ['time', 'el']].copy()
        obs_flank['el'] = obs_flank['el'] + el_spread*np.random.randn(obs_flank.shape[0])

        # Only keep every dt_scale'th observation and reindex - dt_scale=5 corrsponds to ~2hr
        obs_flank = obs_flank[::cadence]
        obs_flank.set_index(np.arange(0, obs_flank.shape[0]), inplace=True)

        # Only return up to el_max ~ (approx HI1 FOV is 25deg)
        id_fov = (obs_flank['el'] >= el_min) & (obs_flank['el'] <= el_max)
        obs_flank = obs_flank[id_fov]
        # Reindex to start from 0, or loops misbehave.
        obs_flank.set_index(np.arange(0, obs_flank.shape[0]), inplace=True)
        return obs_flank
    

def setup_huxt(dt_scale):
    """
    Initialise HUXt with some predetermined boundary/initial conditions
    Here a uniform 400km/s wind is used, and HUXt time is set to 2008-01-01T00:00:00.
    :param dt_scale: Scalar specifying cadence of output timesteps
    :return:
    """
    start_time = Time('2008-01-01T00:00:00')
    cr_num = np.fix(sn.carrington_rotation_number(start_time))
    ert = H.Observer('EARTH', start_time)
    vr_in = np.zeros(128) + 400*u.km/u.s
    
    # Set up HUXt for a 3.5 day simulation, outputting every dt_scale 
    model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
                   lon_start=290*u.deg, lon_stop=70*u.deg, simtime=2*u.day, dt_scale=dt_scale)
    
    model1d = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
                   lon_out=0*u.deg, simtime=5*u.day, dt_scale=4)
    
    return model, model1d


def get_base_cme(v=1000, lon=0, lat=0, width=35, thickness=1):
    """
    Return the base CME, which is used to establish the pseudo-truth CME and the SIR ensemble
    :param v: CME speed in km/s
    :param lon: CME longtiude in degrees
    :param lat: CME latitude in degrees
    :param width: CME width in degrees
    :param thickness: CME thickness in solar radii
    :return:
    """
    t_launch = (1*u.hr).to(u.s)
    cme = H.ConeCME(t_launch=t_launch, longitude=lon*u.deg, latitude=lat*u.deg, width=width*u.deg, v=v*(u.km/u.s),
                    thickness=thickness*u.solRad)
    return cme


def get_cme_scenario(model, scenario):
    """
    Return the base CME, which is used to establish the pseudo-truth CME and the SIR ensemble
    :param model: A HUXt instance.
    :param scenario: String, 'Average', 'Fast', or 'Extreme'
    :return: ConeCME object
    """
    scenarios = load_cme_scenarios()
    
    v = scenarios[scenario]['speed']
    width = scenarios[scenario]['width']
    lat = model.latitude.to(u.deg)
    t_launch = (1*u.hr).to(u.s)
    cme = H.ConeCME(t_launch=t_launch, longitude=0*u.deg, latitude=lat, width=width, v=v,
                    thickness=1*u.solRad)
    return cme


def perturb_cme(cme):
    """
    Perturb a ConeCME's parameters according to each parameters perturbation function.
    Current version only perturbs CME initial speed.
    :param cme: A ConeCME object
    :return cme_perturb: A ConeCME object with perturbed parameters
    """
    v_new = perturb_cme_speed(cme.v)
    width_new = perturb_cme_width(cme.width)
    lon_new = perturb_cme_width(cme.longitude)
    
    cme_perturb = H.ConeCME(t_launch=cme.t_launch,
                            longitude=lon_new,
                            latitude=cme.latitude,
                            width=width_new,
                            v=v_new,
                            thickness=cme.thickness)
    return cme_perturb


def perturb_cme_longitude(lon):
    """
    Perturb a CMEs source longitude. Perturbation drawn from uniform distribution with prescribed spread.
    :param lon: CME source longitude in degrees.
    :return lon_out: Perturbed CME source longitude in degrees.
    """
    lon_spread = 5*u.deg
    lon_out = lon + np.random.uniform(-1, 1, 1)*lon_spread
    
    # Check lon_out is between 0 and 2 pi.
    lon_out = H._zerototwopi_(lon_out.to(u.rad).value)
    lon_out = lon_out*u.rad
    return lon_out[0]


def perturb_cme_latitude(lat):
    """
    Perturb a CMEs source latitude. Perturbation drawn from uniform distribution with prescribed spread.
    :param lat: CME source latitude in degrees.
    :return lat_out: Perturbed CME source latitude in degrees.
    """
    lat_spread = 5*u.deg
    lat_out = lat + np.random.uniform(-1, 1, 1)*lat_spread
    return lat_out[0]


def perturb_cme_speed(speed):
    """
    Perturb a CMEs initial speed in km/s. Perturbation drawn from uniform distribution with prescribed spread.
    :param speed: CME initial speed in km/s.
    :return speed_out: Perturbed CME initial speed in km/s.
    """
    speed_spread = 0.1 * speed #set speed spread to 10% of speed
    speed_out = speed + np.random.uniform(-1, 1, 1)*speed_spread
    return speed_out[0]


def perturb_cme_width(width):
    """
    Perturb a CMEs angular width.. Perturbation drawn from uniform distribution with prescribed spread. 
    :param width: CME angular width in degrees.
    :return width_out: Perturbed CME angular width in degrees.
    """
    width_spread = 5*u.deg
    width_out = width + np.random.uniform(-1, 1, 1)*width_spread
    return width_out[0]


def generate_cme_ensemble(cme, n_ensemble):
    """
    Function to an ensemble of ConeCMEs.
    :param cme: A ConeCME instance.
    :param n_ensemble: The number of ensemble members to generate
    :return cme_ensemble: A list of ConeCME instances generated by perturbing the input cme
    """
    cme_ensemble = []
    for i in range(n_ensemble):
        cme_perturbed = perturb_cme(cme)
        cme_ensemble.append(cme_perturbed)
        
    return cme_ensemble
    

def open_SIR_output_file(output_dir, tag):
    """
    Function to open a HDF5 file to store the SIR analysis steps.
    :param output_dir: The absolute path to the directory to store the analysis file
    :param tag: A string tag to append to the file name
    :return file: The file object for controlling data I/O.
    :return filepath: String path of the output file
    """
    name = 'SIR_HUXt_{}.hdf5'.format(tag)
    if os.path.isdir(output_dir):
        filepath = os.path.join(output_dir, name)
    else:
        dirs = get_project_dirs()
        filepath = os.path.join(dirs['sir_analysis'], name)
        
    file = h5py.File(filepath, 'w')
    
    return file, filepath


def initialise_cme_parameter_ensemble_arrays(n_ensemble):
    """
    Function to initialise empty arrays for storing the CME parameters for each ensemble member at each analysis step
    :param n_ensemble: The number of ensemble members in the SIR analysis
    :return parameter_arrays: A dictionary of parameter keys and an empty array for storing each ensemble member value
    """
    keys = ['t_init', 'v', 'width', 'lon', 'lat', 'thick', 't_transit', 'v_hit', 'likelihood', 'weight']
    parameter_arrays = {k:np.zeros(n_ensemble) for k in keys}
    parameter_arrays['n_members'] = n_ensemble
    return parameter_arrays


def update_analysis_file_initial_values(file_handle, cme, observations, deterministic_run):
    """
    Function to output the modelled CME initial values and the CME observations to the SIR analysis file
    :param file_handle: The HDF5 file object of the analysis file.
    :param cme: A ConeCME instance representing the best guess inital CME parameter values.
    :param observations: Dictionary containing observations of the true CME
    :param deterministic_run: Dictionary containing arrival time and speed stats for deterministic run
                             of best-guess CME conditions
    """
    
    file_handle.create_dataset('cme_inital_values', data=cme.parameter_array())
    file_handle.create_dataset('truth_cme_params', data=observations['truth_cme_params'])
    file_handle.create_dataset('t_transit_truth', data=observations['t_transit'])
    file_handle.create_dataset('v_hit_truth', data=observations['v_hit'])
    file_handle.create_dataset('observer_lon', data=observations['observer_lon'].value)
    file_handle.create_dataset('observed_cme', data=observations['observed_cme_flank'])
    file_handle.create_dataset('t_transit_det', data=deterministic_run['t_transit'])
    file_handle.create_dataset('v_hit_det', data=deterministic_run['v'])
    keys = observations['observed_cme_flank'].columns.to_list()
    col_names = "    ".join(keys)
    file_handle.create_dataset('observed_cme_keys', data=col_names)
    file_handle.flush()
    return


def update_analysis_file_ensemble_members(analysis_group, parameter_array, ens_profiles):
    """
    Function to output the ensemble of CME paramters, likelihoods, weights, and time-elongation profiles at this analysis step.
    :param parameter_array: Dictionary of arrays containing the CME parameters, likelihoods and weights
    :param ens_profiles: Pandas dataframe containing the time-elongation profiles for each ensemble member at this analysis step.
    """
    # Save ensemble member parameters to file
    analysis_group.create_dataset('t_launch', data=parameter_array['t_init'])
    analysis_group.create_dataset('v', data=parameter_array['v'])
    analysis_group.create_dataset('lon', data=parameter_array['lon'])
    analysis_group.create_dataset('lat', data=parameter_array['lat'])
    analysis_group.create_dataset('width', data=parameter_array['width'])
    analysis_group.create_dataset('thick', data=parameter_array['thick'])
    analysis_group.create_dataset('t_transit', data=parameter_array['t_transit'])
    analysis_group.create_dataset('v_hit', data=parameter_array['v_hit'])
    analysis_group.create_dataset('likelihood', data=parameter_array['likelihood'])
    analysis_group.create_dataset('weight', data=parameter_array['weight'])
    analysis_group.create_dataset('ens_profiles', data=ens_profiles)
    keys = ens_profiles.columns.to_list()
    col_names = "    ".join(keys)
    analysis_group.create_dataset('ens_profiles_keys', data=col_names)
    return


def compute_observation_likelihood(t_obs, e_obs, model_flank):
    """
    Function to compute the likelihood of obtaining the observed CME flank elongation, given the modelled flank elongation.
    Assumes a Gaussian likelihood function with fixed variance. 
    :param t_obs: Float value of the time of the observed elongation point
    :param e_obs: Float value of the observed elongation point
    :param model_flank: A pandas dataframe containing the time-elongation profile of the modelled CME flank, the model_flank attribute of a
                        ConeCME object.
    :return likelihood: Float value of the likelihood of these observations.
    """
    # Find closest time match between observed and modelled flank
    # There should be an exact match, but this is safer
    id_obs = np.argmin(np.abs(model_flank['time'].values - t_obs))
    # Get modelled elongation at closest match
    e_member = model_flank.loc[id_obs, 'el']
    # Compute likelihood of obs given modelled flank using Gaussian likelihood function
    likelihood = st.norm.pdf(e_obs, loc=e_member, scale=0.15)
    
    return likelihood

    
def zscore(x):
    """
    Compute z-scores of a parameter
    : param x: An array of values
    : return x_z: An array of Z scores of x
    : return x_avg: The mean of x
    : return x_std: The standard of deviation of x
    """
    x_avg = np.mean(x)
    x_std = np.std(x)
    x_z = (x - x_avg) / x_std
    return x_z, x_avg, x_std


def anti_zscore(x_z, x_avg, x_std):
    """
    Compute the inverse zscore transform to go from a zscore back to a state variable.
    : param x_z: An array of Z scores of x
    : param x_avg: The mean of x
    : param x_std: The standard of deviation of x
    : return x: An array of values
    """
    x = x_std*x_z + x_avg
    return x


def compute_resampling(parameter_array):
    """
    Use gaussian kernel density estimation to generate new cme parameter values from the weighted distribution of
    current values. Current version only resamples on CME speed.
    :param parameter_arrays: Dictionary of arrays containing the CME parameters, likelihoods and weights
    :return resampled_cmes: A list of ConeCME objects initialised with the resampled CME parameters
    """
    
    v = parameter_array['v']
    width = parameter_array['width']
    lon = parameter_array['lon']
    weights = parameter_array['weight']
    
    lon[lon>180] -= 360
    
    # Pull out paramters from dict
    id_good = np.isfinite(weights)
    weights = weights[id_good]
    v = v[id_good]
    width = width[id_good]
    lon = lon[id_good]

    # Convert speeds to z-scores
    v_z, v_avg, v_std = zscore(v)
    width_z, width_avg, width_std = zscore(width)
    lon_z, lon_avg, lon_std = zscore(lon)

    data = np.array([v_z, width_z, lon_z]).T

    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data, sample_weight=weights.ravel())

    # Resample the particles, and convert back to parameter space from zscore
    n_members = parameter_array['n_members']
    resample = kde.sample(n_members)

    v_z_resamp =  resample[:, 0]
    v_resamp = anti_zscore(v_z_resamp, v_avg, v_std)

    width_z_resamp =  resample[:, 1]
    width_resamp = anti_zscore(width_z_resamp, width_avg, width_std)
    
    lon_z_resamp =  resample[:, 2]
    lon_resamp = anti_zscore(lon_z_resamp, lon_avg, lon_std)

    id_bad_v = v_resamp <= 0
    if np.any(id_bad_v):
        print("Warning: Negative (unphysical) speed samples. Reflecting")
        v_resamp[id_bad_v] = np.abs(v_resamp[id_bad_v])

    id_bad_width = width_resamp <= 0
    if np.any(id_bad_width):
        print("Warning: Negative (unphysical) width samples. Reflecting")
        width_resamp[id_bad_width] = np.abs(width_resamp[id_bad_width])

    # Make new list of ConeCMEs from resampled parameters
    resampled_cmes = []

    # Get initiation and thickness parameters, as these are fixed.
    t_init = parameter_array['t_init']
    thick = parameter_array['thick']
    lat = parameter_array['lat']

    for i in range(n_members):
        # Resampled params
        v_new = v_resamp.ravel()[i]*(u.km/u.s)
        width_new = width_resamp.ravel()[i]*u.deg
        lon_new = lon_resamp.ravel()[i]*u.deg
        # Fixed params
        t_launch = t_init[i]*u.s
        lat_new = lat[i]*u.deg
        thickness = thick[i]*u.solRad
        conecme = H.ConeCME(t_launch=t_launch, longitude=lon_new, latitude=lat_new, width=width_new, v=v_new, thickness=thickness)
        resampled_cmes.append(conecme)

    return resampled_cmes    
    
    
def SIR(model, model1d, cme, observations, n_ens, output_path, tag):
    """
    Function implementing the Sequential Importance Resampling of initial CME parameters in HUXt
    :param model: A HUXt instance.
    :param cme: A ConeCME instance representing the best guess inital CME parameter values.
    :param observations: A dictionary containing the observer longitude, a pandas data frame of the
                        observed CME flank elongation, and an array of the truth CME parameters
    :param n_ens: The number of ensemble members.
    :param output_path: The full path of a directory to store the SIR analysis file
    :param tag: A string to append to output file name.
    """
    
    # Define constants of the SIR scheme.
    n_analysis_steps = 8
    
    # Open file for storing SIR analysis
    out_file, out_filepath = open_SIR_output_file(output_path, tag)
    
    observed_cme = observations['observed_cme_flank']
    observer_lon = observations['observer_lon']
    
    # Compute deterministic run
    model1d.solve([cme])
    cme_det = model1d.cmes[0]
    deterministic_run = cme_det.compute_arrival_at_body('EARTH')
    deterministic_run['t_transit'] = deterministic_run['t_transit'].to(u.s).value
    deterministic_run['v'] = deterministic_run['v'].value
    
    # Output the initial CME and observation data
    update_analysis_file_initial_values(out_file, cme, observations, deterministic_run)
    
    # Generate the initial ensemble 
    cme_ensemble = generate_cme_ensemble(cme, n_ens)
    
    # Loop through the observations for each analysis step
    for i in range(n_analysis_steps):
        
        # Set up group to store this analysis step
        analysis_key = "analysis_{:02d}".format(i)
        analysis_group = out_file.create_group(analysis_key)
        
        # Get the observed CME flank time-elongation point
        t_obs = observed_cme.loc[i, 'time']
        e_obs = observed_cme.loc[i, 'el']
        
        # Output the observations to file
        analysis_group.create_dataset('t_obs', data=t_obs)
        analysis_group.create_dataset('e_obs', data=e_obs)
        
        #Preallocate space for CME parameters for each ensemble member
        parameter_array = initialise_cme_parameter_ensemble_arrays(n_ens)

        # Loop through the ensemble members and compare to observations.
        for j in range(n_ens):
            
            # Run HUXt using this cme ensemble member
            model.solve([cme_ensemble[j]])
            cme_member = model.cmes[0]
            
            # Also do 1D hi-res run for arrival time calculations
            model1d.solve([cme_ensemble[j]])
            cme_arr = model1d.cmes[0]
            arrival_stats = cme_arr.compute_arrival_at_body('EARTH')
            t_transit = arrival_stats['t_transit']
            v_hit = arrival_stats['v']
            
            # Update CME parameter arrays
            parameter_array['t_init'][j] = cme_member.t_launch.value
            parameter_array['v'][j] = cme_member.v.value
            parameter_array['lon'][j] = cme_member.longitude.to(u.deg).value
            parameter_array['lat'][j] = cme_member.latitude.to(u.deg).value
            parameter_array['width'][j] = cme_member.width.to(u.deg).value
            parameter_array['thick'][j] = cme_member.thickness.to(u.solRad).value
            parameter_array['t_transit'][j] = t_transit.to(u.s).value
            parameter_array['v_hit'][j] = v_hit.value
            
            # Get pseudo-observations of this ensemble member
            member_obs = Observer(model, cme_member, observer_lon) 
            
            # Plot out the ensemble member
            #fig, ax = sirplt.plot_huxt_with_observer(model.time_out[8], model, [member_obs], add_flank=True, add_fov=True)
            #fig.savefig(tag + "a{:02d}_e{:02d}.png".format(i, j))
            #plt.close('all')
            
            # Compute the likelihood of the observation given this members time-elongation profile
            parameter_array['likelihood'][j] = compute_observation_likelihood(t_obs, e_obs, member_obs.model_flank)
            
            # Collect all the ensemble time-elongation profiles together. 
            if j == 0: 
                ens_profiles = member_obs.model_flank.copy()
                ens_profiles.drop(columns=['r', 'lon'], inplace=True)
                ens_profiles.rename(columns={'el': 'e_{:02d}'.format(j)}, inplace=True)
            else:
                ens_profiles['e_{:02d}'.format(j)] = member_obs.model_flank['el'].copy()
         
        # Compute particle weights from likelihoods
        parameter_array['weight'] = parameter_array['likelihood'] / np.nansum(parameter_array['likelihood'])
                
        # Update the output file with the ensemble parameters, weights, and time-elongation profiles at this analysis step
        update_analysis_file_ensemble_members(analysis_group, parameter_array, ens_profiles)
        
        # Resample the particles based on the current weights.
        cme_ensemble = compute_resampling(parameter_array)
        
        # Push data to the file
        out_file.flush()
        
    out_file.close()
        
    return


def get_project_dirs():
    """
    Function to pull out the directories of boundary conditions, ephemeris, and to save figures and output data.
    """
    
    # Get path of huxt.py, and work out root dir of HUXt repository
    cwd = os.path.abspath(os.path.dirname(__file__))
    root = os.path.dirname(cwd)
   
    # Config file must be saved in HUXt/code
    config_file = os.path.join(cwd, 'config.dat')
    
    if os.path.isfile(config_file):
        
        with open(config_file, 'r') as file:
            lines = file.read().splitlines()
            dirs = {line.split(',')[0]: os.path.join(root, line.split(',')[1]) for line in lines}
            
        dirs['root'] = root

        # Just check the directories exist.
        for key, val in dirs.items():
                if key in ['ephemeris', 'HELCATS_data']:
                    if not os.path.isfile(val):
                        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), val)
                else:
                    if not os.path.isdir(val):
                        raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), val)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)
                
    return dirs


def build_cme_scenarios():
    """
    Function to build the average, fast, and extreme CME scenarios used in the modelling.
    These are built from the percentiles of CME speed and width distributions of the 
    HELCATS GCS fits in WP3 KINCAT (https://www.helcats-fp7.eu/catalogues/wp3_kincat.html)
    """
    project_dirs = get_project_dirs()
    column_names = ['ID', 'pre_date', 'pre_time', 'last_date', 'last_time', 'carlon',
                    'stolon', 'stolat', 'tilt', 'ssprat', 'h_angle', 'speed',  'mass']
    data = pd.read_csv(project_dirs['HELCATS_data'], names=column_names, skiprows=4, delim_whitespace=True)

    speed_percentiles = {'slow': 0.5, 'fast': 0.85}
    width_percentiles = {'narrow': 0.5, 'wide': 0.85}

    # Setup output file. Overwrite if exists.
    out_filepath = os.path.join(project_dirs['out_data'], "CME_scenarios.hdf5")
    if os.path.isfile(out_filepath):
        print("Warning: {} already exists. Overwriting".format(out_filepath))
        os.remove(out_filepath)

    out_file = h5py.File(out_filepath, 'w')

    for vkey, vp in speed_percentiles.items():
        speed = data['speed'].quantile(vp)

        for wkey, wp in width_percentiles.items():

            width = 2*data['h_angle'].quantile(wp)

            cme_group = out_file.create_group("{}_{}".format(vkey, wkey))
            cme_group.create_dataset('speed_percentile', data=vp)                              
            cme_group.create_dataset('width_percentile', data=wp)                              
            dset = cme_group.create_dataset('speed', data=speed)
            dset.attrs['unit'] = (u.km/u.s).to_string()
            dset = cme_group.create_dataset('width', data=width)
            dset.attrs['unit'] = u.deg.to_string()


    out_file.close()
    return


def load_cme_scenarios():
    """
    Load in the CME scenarios from their HDF5 file and return them in a dictionary.
    """
    
    project_dirs = get_project_dirs()
    datafile_path = os.path.join(project_dirs['out_data'], 'CME_scenarios.hdf5')
    datafile = h5py.File(datafile_path, 'r')
    cme_scenarios = {}
    for key in datafile.keys():
        cme = datafile[key]
        speed = cme['speed'][()] * u.Unit(cme['speed'].attrs['unit'])
        width = cme['width'][()] * u.Unit(cme['width'].attrs['unit'])
        cme_scenarios[key] = {'speed': speed, 'width': width}

    datafile.close()
    
    return cme_scenarios