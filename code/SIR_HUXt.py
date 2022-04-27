from astropy.time import Time
import astropy.units as u
import h5py
import numpy as np
import os
import pandas as pd
import sunpy.coordinates.sun as sn
import scipy.stats as st
from sklearn.neighbors import KernelDensity
# Local packages
import huxt as H
import huxt_inputs as Hin
import huxt_analysis as Ha


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

            # Find the flank coordinate and update output
            id_obs_flank = np.argmax(e_obs)       
            flank.loc[i, 'lon'] = lon_cme[id_obs_flank].value
            flank.loc[i, 'r'] = r_cme[id_obs_flank].value
            flank.loc[i, 'el'] = np.rad2deg(e_obs[id_obs_flank])

        # Force values to be floats.
        keys = ['lon', 'r', 'el']
        flank[keys] = flank[keys].astype(np.float64)
        return flank
    
    
    def compute_synthetic_obs(self, el_spread=0.5, cadence=5, el_min=4.0, el_max=30.0):
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
    

def setup_huxt(start_time, uniform_wind=True):
    """
    Initialise HUXt with some predetermined boundary/initial conditions
    uniform_wind is flag for setting uniform 400km/s wind.
    :param start_time: An astropy.Time object specifying the start time of HUXt
    :param uniform_wind: If True, set the wind to be uniform 400km/s
    :return:
    """
    cr_num = np.fix(sn.carrington_rotation_number(start_time))
    ert = H.Observer('EARTH', start_time)

    # Set up HUXt for a 5 day simulation of this CR
    vr_in = Hin.get_MAS_long_profile(cr_num, ert.lat.to(u.deg))
    # Set wind to be uniform?
    if uniform_wind:
        vr_in = np.zeros(vr_in.shape) + 400*vr_in.unit
        
    model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
                   lon_start=300*u.deg, lon_stop=60*u.deg, simtime=3.5*u.day, dt_scale=4)
    
    return model


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


def perturb_cme(cme):
    """
    Perturb a ConeCME's parameters according to each parameters perturbation function.
    Current version only perturbs CME initial speed.
    :param cme: A ConeCME object
    :return cme_perturb: A ConeCME object with perturbed parameters
    """

    v_new = perturb_cme_speed(cme.v)
    
    cme_perturb = H.ConeCME(t_launch=cme.t_launch,
                            longitude=cme.longitude,
                            latitude=cme.latitude,
                            width=cme.width,
                            v=v_new,
                            thickness=cme.thickness)
    return cme_perturb


def perturb_cme_longitude(lon):
    """
    Perturb a CMEs source longitude. Perturbation drawn from uniform distribution with prescribed spread.
    :param lon: CME source longitude in degrees.
    :return lon_out: Perturbed CME source longitude in degrees.
    """
    lon_spread = 10*u.deg
    lon_out = lon + np.random.uniform(-1,1,1)*lon_spread
    return lon_out[0]


def perturb_cme_latitude(lat):
    """
    Perturb a CMEs source latitude. Perturbation drawn from uniform distribution with prescribed spread.
    :param lat: CME source latitude in degrees.
    :return lat_out: Perturbed CME source latitude in degrees.
    """
    lat_spread = 10*u.deg
    lat_out = lat + np.random.uniform(-1,1,1)*lat_spread
    return lat_out[0]


def perturb_cme_speed(speed):
    """
    Perturb a CMEs initial speed in km/s. Perturbation drawn from uniform distribution with prescribed spread.
    :param speed: CME initial speed in km/s.
    :return speed_out: Perturbed CME initial speed in km/s.
    """
    speed_spread = 50*(u.km/u.s)
    speed_out = speed + np.random.uniform(-1,1,1)*speed_spread
    return speed_out[0]


def perturb_cme_width(width):
    """
    Perturb a CMEs angular width.. Perturbation drawn from uniform distribution with prescribed spread. 
    :param width: CME angular width in degrees.
    :return width_out: Perturbed CME angular width in degrees.
    """
    width_spread = 10*u.deg
    width_out = width + np.random.uniform(-1,1,1)*width_spread
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
    

def open_SIR_output_file(tag):
    """
    Function to open a HDF5 file to store the SIR analysis steps.
    :param tag: A string tag to append to the file name
    :return out_file: The file object for controlling data I/O.
    :return out_filepath: String path of the output file
    """
    out_filepath = '/home/users/yq904481/research/repos/SIR_HUXt/data/out_data/SIR_HUXt_{}.hdf5'.format(tag)
    out_file = h5py.File(out_filepath, 'w')
    
    return out_file, out_filepath


def initialise_cme_parameter_ensemble_arrays(n_ensemble):
    """
    Function to initialise empty arrays for storing the CME parameters for each ensemble member at each analysis step
    :param n_ensemble: The number of ensemble members in the SIR analysis
    :return:
    """
    speeds = np.zeros(n_ensemble)
    widths = np.zeros(n_ensemble)
    lons = np.zeros(n_ensemble)
    lats = np.zeros(n_ensemble)
    thicks = np.zeros(n_ensemble)
    arrivals = np.zeros(n_ensemble)
    likelihood = np.zeros(n_ensemble)
    return speeds, widths, lons, lats, thicks, arrivals, likelihood


def update_analysis_file_initial_values(file_handle, cme, observations):
    """
    Function to output the modelled CME initial values and the CME observations to the SIR analysis file
    :param file_handle: The HDF5 file object of the analysis file.
    :param cme: A ConeCME instance representing the best guess inital CME parameter values.
    :param observed_cme: A pandas dataframe of observations of the CME time elongation profile.
    :param observer_lon: The longitude of the observer relative to Earth, in degrees.
    """
    
    file_handle.create_dataset('cme_inital_values', data=cme.parameter_array())
    file_handle.create_dataset('t_arrive', data=observations['t_arrive'].jd)
    file_handle.create_dataset('t_transit', data=observations['t_transit'].value)
    file_handle.create_dataset('cme_params', data=observations['cme_params'])
    file_handle.create_dataset('observer_lon', data=observations['observer_lon'].value)
    file_handle.create_dataset('observed_cme', data=observations['observed_cme_flank'])
    keys = observations['observed_cme_flank'].columns.to_list()
    col_names = "    ".join(keys)
    file_handle.create_dataset('observed_cme_keys', data=col_names)
    file_handle.flush()
    return


def update_analysis_file_ensemble_members(analysis_group, speeds, widths, lons, lats, thicks, arrivals, likelihood, weights, ens_profiles):
    """
    Function to output the ensemble of CME paramters, likelihoods, weights, and time-elongation profiles at this analysis step.
    :param speeds: Array of the ensemble of CME speeds.
    :param widths: Array of the ensemble of CME widths.
    :param lons: Array of the ensemble of CME longitudes.
    :param lats: Array of the ensemble of CME latitudes.
    :param thicks: Array of the ensemble of CME thicknesses.
    :param arrivals: Array of the ensemble of CME arrival times at Earth.
    :param likelihood: Array of the ensemble of likelihoods of the observed CME elongation given the modelled ensemble at this analysis step.
    :param weights: Array of the particle weights derived from the likelihoods.
    :param ens_profiles: Pandas dataframe containing the time-elongation profiles for each ensemble member at this analysis step.
    """
    # Save ensemble member parameters to file
    analysis_group.create_dataset('speeds', data=speeds)
    analysis_group.create_dataset('lons', data=lons)
    analysis_group.create_dataset('lats', data=lats)
    analysis_group.create_dataset('widths', data=widths)
    analysis_group.create_dataset('thicks', data=thicks)
    analysis_group.create_dataset('arrivals', data=arrivals)
    analysis_group.create_dataset('likelihood', data=likelihood)
    analysis_group.create_dataset('weights', data=weights)
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
    likelihood = st.norm.pdf(e_obs, loc=e_member, scale=0.2)
    
    return likelihood

    
def compute_resampling(speeds, lons, lats, widths, thicks, weights):
    """
    Use gaussian kernel density estimation to generate new cme parameter values from the weighted distribution of
    current values. Current version only resamples on CME speed.
    :param speeds: Array of CME speeds of current ensemble members (in km/s)
    :param lons: Array of CME longitudes of current ensemble members (in degs)
    :param lats: Array of CME latitudes of current ensemble members (in degs)
    :param widths: Array of CME widths of current ensemble members (in degs)
    :param thicks: Array of CME thicknesses of current ensemble members (in solRad)
    :param weights: Array of weights of the current ensemble members
    :return resampled_cmes: A list of ConeCME objects initialised with the resampled CME parameters
    """
    
    n_members = speeds.size
    
    # Remove any particles with invalid weights
    id_good = np.isfinite(weights)
    weights = weights[id_good]
    v = speeds[id_good]
    
    # Convert speeds to z-scores
    v_av = np.mean(v)
    v_std = np.std(v)
    v_z = (v - v_av) / v_std 
    
    # Weighted Gaussian KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(v_z.reshape(-1,1), sample_weight=weights.ravel())
    
    # Get resampled particle speeds
    v_z_resample = kde.sample(n_members)
    # Invert z-scores to real space
    v_resample = v_z_resample*v_std + v_av
    
    # Make new list of ConeCMEs from resampled parameters
    resampled_cmes = []
    
    # Get launch time from base CME, as this is fixed
    base_cme = get_base_cme()
    t_launch = base_cme.t_launch
    for i in range(n_members):
        v_new = v_resample.ravel()[i]*(u.km/u.s)
        lon = lons[i]*u.deg
        lat = lats[i]*u.deg
        width = widths[i]*u.deg
        thickness = thicks[i]*u.solRad
        conecme = H.ConeCME(t_launch=t_launch, longitude=lon, latitude=lat, width=width, v=v_new, thickness=thickness)
        resampled_cmes.append(conecme)
        
    return resampled_cmes

    
def SIR(model, cme, observations, n_ens, tag):
    """
    Function implementing the Sequential Importance Resampling of initial CME parameters in HUXt
    :param model: A HUXt instance.
    :param cme: A ConeCME instance representing the best guess inital CME parameter values.
    :param observations: A dictionary containing the observed CME arrival time, transit time, observer longitude and a
                         pandas data frame of the observed CME flank elongation.
    :param n_ens: The number of ensemble members.
    :param tag: A string to append to output file name.
    """
    
    # Define constants of the SIR scheme.
    n_analysis_steps = 8
    
    # Open file for storing SIR analysis
    out_file, out_filepath = open_SIR_output_file(tag)
    
    observed_cme = observations['observed_cme_flank']
    observer_lon = observations['observer_lon']
    
    # Output the initial CME and observation data
    update_analysis_file_initial_values(out_file, cme, observations)
    
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
        speeds, widths, lons, lats, thicks, arrivals, likelihood = initialise_cme_parameter_ensemble_arrays(n_ens)

        # Loop through the ensemble members and compare to observations.
        for j in range(n_ens):
            
            # Run HUXt using this cme ensemble member
            model.solve([cme_ensemble[j]])
            
            cme_member = model.cmes[0]
            
            
            # Update CME parameter arrays
            speeds[j] = cme_member.v.value
            lons[j] = cme_member.longitude.to(u.deg).value
            lats[j] = cme_member.latitude.to(u.deg).value
            widths[j] = cme_member.width.to(u.deg).value
            thicks[j] = cme_member.thickness.to(u.solRad).value
            hit, t_arrive, t_transit, hit_lon, hit_id = cme_member.compute_arrival_at_body('EARTH')
            arrivals[j] = t_arrive.jd
            
            # Get pseudo-observations of this ensemble member
            member_obs = Observer(model, cme_member, observer_lon) 
            # Compute the likelihood of the observation given this members time-elongation profile
            likelihood[j] = compute_observation_likelihood(t_obs, e_obs, member_obs.model_flank)
            
            # Collect all the ensemble time-elongation profiles together. 
            if j == 0: 
                ens_profiles = member_obs.model_flank.copy()
                ens_profiles.drop(columns=['r', 'lon'], inplace=True)
                ens_profiles.rename(columns={'el': 'e_{:02d}'.format(j)}, inplace=True)
            else:
                ens_profiles['e_{:02d}'.format(j)] = member_obs.model_flank['el'].copy()
         
        # Compute particle weights from likelihoods
        weights = likelihood / np.nansum(likelihood)
        
        # Update the output file with the ensemble parameters, weights, and time-elongation profiles at this analysis step
        update_analysis_file_ensemble_members(analysis_group, speeds, widths, lons, lats, thicks, arrivals, likelihood, weights, ens_profiles)
        
        # Resample the particles based on the current weights.
        cme_ensemble = compute_resampling(speeds, lons, lats, widths, thicks, weights)
        
        # Push data to the file
        out_file.flush()
        
    out_file.close()
        
    return
