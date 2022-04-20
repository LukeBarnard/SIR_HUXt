from astropy.time import Time
import astropy.units as u
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import os
import pandas as pd
import sunpy.coordinates.sun as sn
import scipy.stats as st
# Local packages
import huxt as H
import huxt_inputs as Hin
import huxt_analysis as Ha

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
                   lon_start=270*u.deg, lon_stop=90*u.deg, simtime=3.5*u.day, dt_scale=4)
    
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


def perturb_cone_cme(cme):
    """
    Perturb a ConeCME's parameters. Used to establish the pseudo-truth CME and the initial SIR ensemble members.
    :param cme: A ConeCME object
    :return:
    """
    lon_spread = 10*u.deg
    lat_spread = 10*u.deg
    width_spread = 10*u.deg
    v_spread = 150*(u.km/u.s)
    thickness_spread = 1*u.solRad
    
    randoms = np.random.uniform(-1, 1, 5)
    lon_new = cme.longitude + randoms[0]*lon_spread
    lat_new = cme.latitude + randoms[1]*lat_spread
    width_new = cme.width + randoms[2]*width_spread
    v_new = cme.v + randoms[3]*v_spread
    thickness_new = cme.thickness + randoms[4]*thickness_spread
    
    cme_perturb = H.ConeCME(t_launch=cme.t_launch,
                            longitude=lon_new,
                            latitude=lat_new,
                            width=width_new,
                            v=v_new,
                            thickness=thickness_new)
    return cme_perturb


class Observer:
    
    @u.quantity_input(longitude=u.deg)
    def __init__(self, model, longitude, el_min=4.0, el_max=30.0):
        
        ert_ephem = model.get_observer('EARTH')
        
        self.time = ert_ephem.time 
        self.r = ert_ephem.r
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
        
        cme = model.cmes[0]
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
        Return synthetic observations with a specified uncertainty spread, cadence, and maximum elongation.
        el_spread = standard deviation of random gaussian noise added to the modelled elongation.
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


def plot_huxt_with_observer(time, model, observer, add_flank=False, add_fov=False):
    """
    Plot the HUXt solution at a specified time, and (optionally) overlay the modelled flank location and field of view
    of a specified observer.
    :param time: The time to plot. The closest value in model.time_out is selected.
    :param model: A HUXt instance with the solution in.
    :param observer: An Observer instance with the modelled flank.
    :param add_flank: If True, add the modelled flank.
    :param add_fov: If True, highlight the observers field of view.
    :return:
    """
    
    id_t = np.argmin(np.abs(model.time_out - time))

    # Get plotting data
    lon_arr, dlon, nlon = H.longitude_grid()
    lon, rad = np.meshgrid(lon_arr.value, model.r.value)
    mymap = mpl.cm.viridis
    v_sub = model.v_grid_cme.value[id_t, :, :].copy()
    # Insert into full array
    if lon_arr.size != model.lon.size:
        v = np.zeros((model.nr, nlon)) * np.NaN
        if model.lon.size != 1:
            for i, lo in enumerate(model.lon):
                id_match = np.argwhere(lon_arr == lo)[0][0]
                v[:, id_match] = v_sub[:, i]
        else:
            print('Warning: Trying to contour single radial solution will fail.')
    else:
        v = v_sub

    # Pad out to fill the full 2pi of contouring
    pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
    lon = np.concatenate((lon, pad), axis=1)
    pad = rad[:, 0].reshape((rad.shape[0], 1))
    rad = np.concatenate((rad, pad), axis=1)
    pad = v[:, 0].reshape((v.shape[0], 1))
    v = np.concatenate((v, pad), axis=1)

    mymap.set_over('lightgrey')
    mymap.set_under([0, 0, 0])
    levels = np.arange(200, 800 + 10, 10)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    cnt = ax.contourf(lon, rad, v, levels=levels, cmap=mymap, extend='both')

    # Add on CME boundaries and Observer
    cme = model.cmes[0]
    ax.plot(cme.coords[id_t]['lon'], cme.coords[id_t]['r'], '-', color='darkorange', linewidth=3, zorder=3)
    ert = model.get_observer('EARTH')
    ax.plot(ert.lon[id_t], ert.r[id_t], 'co', markersize=16, label='Earth')            

    # Add on the observer
    ax.plot(observer.lon[id_t], observer.r[id_t], 's', color='r', markersize=16, label='Observer')
        
    if add_flank:
        flank_lon = observer.model_flank.loc[id_t, 'lon']
        flank_rad = observer.model_flank.loc[id_t, 'r']
        ax.plot(flank_lon, flank_rad, 'r.', markersize=10, zorder=4)
        # Add observer-flank line
        ro = observer.r[id_t]
        lo = observer.lon[id_t]
        ax.plot([lo.value, flank_lon], [ro.value, flank_rad], 'r--', zorder=4)
        
    if add_fov:
        fov_patch = get_fov_patch(observer.r[id_t], observer.lon[id_t], observer.el_min, observer.el_max)
        ax.add_patch(fov_patch)

    ax.set_ylim(0, 240)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.patch.set_facecolor('slategrey')

    fig.subplots_adjust(left=0.05, bottom=0.16, right=0.95, top=0.99)
    # Add color bar
    pos = ax.get_position()
    dw = 0.005
    dh = 0.045
    left = pos.x0 + dw
    bottom = pos.y0 - dh
    wid = pos.width - 2 * dw
    cbaxes = fig.add_axes([left, bottom, wid, 0.03])
    cbar1 = fig.colorbar(cnt, cax=cbaxes, orientation='horizontal')
    cbar1.set_label('Solar Wind speed (km/s)')
    cbar1.set_ticks(np.arange(200, 810, 100))
    return fig, ax


def get_fov_patch(ro, lo, el_min, el_max):
    """
    Function to compute a matplotlib patch to higlight an observers field of view. 
    ro = radius of observer (in solRad)
    lo = longitude of observer (in rad)
    el_min = minimum elongation of the field of view
    el_max = maximum elongation of the field of view
    """
    xo = ro*np.cos(lo)
    yo = ro*np.sin(lo)
    
    fov_patch = [[lo.value, ro.value]]
    
    for el in [el_min, el_max]:

        rp = ro*np.tan(el*u.deg)
        if (lo < 0*u.rad) | (lo > np.pi*u.rad):
            lp = lo + 90*u.deg
        else:
            lp = lo - 90*u.deg

        if lp > 2*np.pi*u.rad:
            lp = lp - 2*np.pi*u.rad

        xp = rp*np.cos(lp)
        yp = rp*np.sin(lp)

        # Wolfram equations for intersection of line with circle
        rf = 475*u.solRad  # set this to a large value outside axis lims so FOV shading spans model domain
        dx = (xp - xo)
        dy = (yp - yo)
        dr = np.sqrt(dx**2 + dy**2)
        det = (xo*yp - xp*yo)
        discrim = np.sqrt((rf*dr)**2 - det**2)

        if (lo < 0*u.rad) | (lo > np.pi*u.rad):
            xf = (det*dy + np.sign(dy)*dx*discrim) / (dr**2)
            yf = (-det*dx + np.abs(dy)*discrim) / (dr**2)
        else:
            xf = (det*dy - np.sign(dy)*dx*discrim) / (dr**2)
            yf = (-det*dx - np.abs(dy)*discrim) / (dr**2)

        lf = np.arctan2(yf, xf)
        fov_patch.append([lf.value, rf.value])

    fov_patch = mpl.patches.Polygon(np.array(fov_patch), color='r', alpha=0.3, zorder=1)
    return fov_patch


def animate_observer(model, obs, tag, add_flank=False, add_fov=False):
    """
    Animate the model solution, and save as an MP4.
    :param model: A HXUt model instance with the solution in
    :param obs: An observer instance containing the modelled flank coords
    :param tag: String to append to filename
    :param add_flank: If True, the modelled flank is plotted
    :param add_fov: If True, the observers field of view is highlighted
    :return:
    """
    # Set the duration of the movie
    # Scaled so a 5 day simulation with dt_scale=4 is a 10 second movie.
    duration = model.simtime.value * (10 / 432000)

    def make_frame(t):
        """
        Produce the frame required by MoviePy.VideoClip.
        :param t: time through the movie
        """
        # Get the time index closest to this fraction of movie duration
        i = np.int32((model.nt_out - 1) * t / duration)
        fig, ax = plot_huxt_with_observer(model.time_out[i], model, obs, add_flank=add_flank, add_fov=add_fov)
        frame = mplfig_to_npimage(fig)
        plt.close('all')
        return frame

    cr_num = np.int32(model.cr_num.value)
    filename = "HUXt_CR{:03d}_{}_movie.mp4".format(cr_num, tag)
    filepath = os.path.join(model._figure_dir_, filename)
    animation = mpy.VideoClip(make_frame, duration=duration)
    animation.write_videofile(filepath, fps=24, codec='libx264')
    return


def plot_elon_profiles_at_analysis(step, model, observer, cme_truth_obs, t_obs, e_obs, ens_profiles, weights):
    """
    Plot the time-elongation profiles of the truth cme, the full ensemble, and the observations.
    :param step: Integer number for the analysis step (used in naming)
    :param model: The HUXt model object
    :param observer: An observer object with the modeled flank
    :param cme_truth_obs: Pandas dataframe of the observed flank of the truth cme
    :param t_obs: Time (in JD) at this analysis step
    :param e_obs: Elongation (in degs) at this analysis step
    :param ens_profiles: Pandas dataframe of the ensemble of time-elongation profiles
    :param weights: An array of the weights of each ensemble member
    :return:
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    keys = ens_profiles.keys()
    keys = keys.drop('time')
    time = (Time(ens_profiles['time'], format='jd') - model.time_init).value*24

    ax[0].plot(time, ens_profiles[keys], '-', color='slategrey', zorder=1, label='Model Ens.')

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=np.nanmin(weights), vmax=np.nanmax(weights))
    for i, w in enumerate(weights):
        key = "e_{:02d}".format(i)
        col = cmap(norm(w))
        time = (Time(ens_profiles['time'], format='jd') - model.time_init).value*24
        ax[1].plot(time, ens_profiles[key], '-', color=col, zorder=1, label='Model Ens.')

    time = (Time(observer.model_flank['time'], format='jd') - model.time_init).value*24
    ax[0].plot(time, observer.model_flank['el'], 'k-', zorder=2, label='Model Truth')

    ax[1].plot(time, observer.model_flank['el'], 'k--', zorder=2, label='Model Truth')

    time = (Time(cme_truth_obs['time'], format='jd') - model.time_init).value*24
    ax[0].plot(time, cme_truth_obs['el'], 'r.', zorder=3, label='Synth. Obs.')

    ax[1].plot(time, cme_truth_obs['el'], 'r.', zorder=3, label='Synth. Obs.')

    time = (t_obs - model.time_init.jd)*24
    ax[1].plot(time, e_obs, 'r*', markersize=10, zorder=3, label='Assimilation point')

    for a in ax:
        a.set_xlabel('Model time (hours)')
        a.set_ylabel('Elongation (deg)')
        # Add legend, remove duplicate labels
        handles, labels = a.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        a.legend(by_label.values(), by_label.keys())

    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position('right')
    ax[1].set_xlim(time - 5, time + 5)
    ax[1].set_ylim(e_obs - 5, e_obs + 5)

    fig.subplots_adjust(left=0.075, bottom=0.12, right=0.925, top=0.98, wspace=0.05)
    fig.savefig('figx_truth_and_ensemble_profiles_with_lkhd_{:02d}.png'.format(step))
    plt.close('all')
    return


def compute_resampling(speeds, lons, lats, widths, thicks, weights):
    """
    Use gaussian kernel density estimation to generate new cme parameter values from the weighted distribution of
    current values
    :param speeds: Array of CME speeds of current ensemble members (in km/s)
    :param lons: Array of CME longitudes of current ensemble members (in degs)
    :param lats: Array of CME latitudes of current ensemble members (in degs)
    :param widths: Array of CME widths of current ensemble members (in degs)
    :param thicks: Array of CME thicknesses of current ensemble members (in solRad)
    :param weights: Array of weights of the current ensemble members
    :return:
    """
    
    n_members = speeds.size
    
    # Remove any bad values
    id_good = np.isfinite(weights)
    weights = weights[id_good]
    speeds = speeds[id_good]
    lons = lons[id_good]
    lats = lats[id_good]
    widths = widths[id_good]
    thicks = thicks[id_good]

    # Make sure longitudes are on -180:180 domain
    lons[lons > 180] -= 360

    params = {'speed': speeds,
              'longitude': lons,
              'latitude': lats,
              'width': widths,
              'thickness': thicks}
    
    samples = {'speed': np.zeros(n_members),
               'longitude': np.zeros(n_members),
               'latitude': np.zeros(n_members),
               'width': np.zeros(n_members),
               'thickness': np.zeros(n_members)}
    
    for i, (key, param) in enumerate(params.items()):

        kde_pos = st.gaussian_kde(param, bw_method=0.175, weights=weights)

        # Resample from posterior for new members.
        new_sample = kde_pos.resample(size=n_members)
        if key == 'thickness':
            # Stop negative thickness
            new_sample[new_sample < 0] = 0.1
                
        samples[key] = new_sample.squeeze()    
            
    # now make a list of cone cme objects using the resampled points. 
    updated_cmes = []
    for i in range(n_members):
        v = samples['speed'][i]*(u.km/u.s)
        lon = samples['longitude'][i]*u.deg
        lat = samples['latitude'][i]*u.deg
        width = samples['width'][i]*u.deg
        thickness = samples['thickness'][i]*u.solRad
        t_launch = (1*u.hr).to(u.s)  # same as the base_cme function
        conecme = H.ConeCME(t_launch=t_launch, longitude=lon, latitude=lat, width=width, v=v, thickness=thickness)
        updated_cmes.append(conecme)
        
    return updated_cmes


def run_experiment():
    np.random.seed(20100114)

    # Set up HUXt with Uniform wind. 
    start_time = Time('2008-06-10T00:00:00')
    model = setup_huxt(start_time, uniform_wind=True)

    # Initialise Earth directed CME. Coords in HEEQ, so need Earth Lat.
    ert = model.get_observer('EARTH')
    avg_ert_lat = np.mean(ert.lat.to(u.deg).value)
    cme_base = get_base_cme(v=1000, lon=0, lat=avg_ert_lat, width=35, thickness=1.1)

    n_truths = 2
    n_members = 50
    observer_lon = -60*u.deg  # approx L5 location

    out_filepath = 'SIR_HUXt_uniform.hdf5'
    out_file = h5py.File(out_filepath, 'w')

    for ttt in range(n_truths):
                    
        truth_key = "truth_{:02d}".format(ttt)
        truth_group = out_file.create_group(truth_key)
        print("{} - {}".format(truth_key, pd.datetime.now().time()))

        # Perturb the base CME to get a "Truth" CME, and solve
        cme_truth = perturb_cone_cme(cme_base)
        model.solve([cme_truth])
        cme_truth = model.cmes[0]

        # Setup an observer at ~L5.
        observer = Observer(model, observer_lon, el_min=10.0, el_max=40.0)
        cme_truth_obs = observer.compute_synthetic_obs(el_spread=0.01, cadence=5, el_min=observer.el_min,
                                                       el_max=observer.el_max)

        # Animate the truth run
        # animate_observer(model, observer, truth_key, add_flank=True, add_fov=True)

        # Save the truth CME parameters and osbervations
        hit, t_arrive, t_transit, hit_lon, hit_id = cme_truth.compute_arrival_at_body('EARTH')
        truth_group.create_dataset('arrival_true', data=t_arrive.jd)
        truth_group.create_dataset('v_true', data=cme_truth.v.value)
        truth_group.create_dataset('lon_true', data=cme_truth.longitude.to(u.deg).value)
        truth_group.create_dataset('lat_true', data=cme_truth.latitude.to(u.deg).value)
        truth_group.create_dataset('width_true', data=cme_truth.width.to(u.deg).value)
        truth_group.create_dataset('thickness_true', data=cme_truth.thickness.to(u.solRad).value)
        truth_group.create_dataset('model_flank_true', data=observer.model_flank.values)
        truth_group.create_dataset('observed_flank', data=cme_truth_obs.values)
        truth_group.create_dataset('n_members', data=n_members)
        truth_group.create_dataset('observer_lon', data=observer_lon.value)

        # Loop through the observations.
        first_pass_flag = True
        for i, row in cme_truth_obs.iterrows():

            analysis_key = "analysis_{:02d}".format(i)
            analysis_group = truth_group.create_group(analysis_key)

            t_obs = row['time']
            e_obs = row['el']

            analysis_group.create_dataset('t_obs', data=t_obs)
            analysis_group.create_dataset('e_obs', data=e_obs)

            speeds = np.zeros(n_members)
            arrivals = np.zeros(n_members)
            lons = np.zeros(n_members)
            lats = np.zeros(n_members)
            widths = np.zeros(n_members)
            thicks = np.zeros(n_members)
            likelihood = np.zeros(n_members)

            for j in range(n_members):

                # Perturb the CME, solve, and get the observer data.
                if first_pass_flag:
                    cme_ens = perturb_cone_cme(cme_base)
                else:
                    cme_ens = updated_cmes[j]

                model.solve([cme_ens])
                cme_ens = model.cmes[0]
                ens_observer = Observer(model, observer_lon, el_min=4.0, el_max=40.0)

                # Collect all the ensemble elongation profiles together. 
                if j == 0: 
                    ens_profiles = ens_observer.model_flank.copy()
                    ens_profiles.drop(columns=['r', 'lon'], inplace=True)
                    ens_profiles.rename(columns={'el': 'e_{:02d}'.format(j)}, inplace=True)
                else:
                    ens_profiles['e_{:02d}'.format(j)] = ens_observer.model_flank['el'].copy()

                # Compute the likelihood of the observation given the members profile
                profile = ens_observer.model_flank.copy()
                # Find closest time - there should be an exact match, but this is safer
                # TODO - add check that closest value isn't too far away?
                id_obs = np.argmin(np.abs(profile['time'].values - t_obs))
                e_mod = profile.loc[id_obs, 'el']
                # Use Gaussian likelihood
                likelihood[j] = st.norm.pdf(e_obs, loc=e_mod, scale=0.2)

                # Save this members CME data
                speeds[j] = cme_ens.v.value
                lons[j] = cme_ens.longitude.to(u.deg).value
                lats[j] = cme_ens.latitude.to(u.deg).value
                widths[j] = cme_ens.width.to(u.deg).value
                thicks[j] = cme_ens.thickness.to(u.solRad).value
                hit, t_arrive, t_transit, hit_lon, hit_id = cme_truth.compute_arrival_at_body('EARTH')
                arrivals[j] = t_arrive.jd

            first_pass_flag = False

            weights = likelihood / np.nansum(likelihood)

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

            out_file.flush()

            # Get resampled CMEs for next iteration
            updated_cmes = compute_resampling(speeds, lons, lats, widths, thicks, weights)

            if i == 8:
                break

    out_file.close()
    return


def run_experiment_random_background():
    np.random.seed(20100114)

    # Set up HUXt with Uniform wind. 
    start_time = Time('2008-06-10T00:00:00')
    end_time = Time('2013-01-05T00:00:00')

    n_truths = 100
    n_members = 50
    observer_lon = -60*u.deg  # approx L5 location

    out_filename = 'SIR_HUXt_multi_truth_randbck_racc.hdf5'
    out_file = h5py.File(out_filename, 'w')
    out_file.create_dataset('n_truths', data=n_truths)

    for ttt in range(n_truths):
        
        initial_time = np.random.uniform(start_time.jd, end_time.jd)
        initial_time = Time(initial_time, format='jd')
        initial_time = Time(initial_time.isot, format='isot') # Why do i need to do this?
        model = setup_huxt(initial_time, uniform_wind=False)

        # Initialise Earth directed CME. Coords in HEEQ, so need Earth Lat.
        ert = model.get_observer('EARTH')
        avg_ert_lat = np.mean(ert.lat.to(u.deg).value)
        cme_base = get_base_cme(v=1000, lon=0, lat=avg_ert_lat, width=35, thickness=1.1)

        truth_key = "truth_{:02d}".format(ttt)
        truth_group = out_file.create_group(truth_key)

        # Perturb the base CME to get a "Truth" CME, and solve
        cme_truth = perturb_cone_cme(cme_base)
        model.solve([cme_truth])
        cme_truth = model.cmes[0]

        # Setup an observer at ~L5.
        observer = Observer(model, observer_lon, el_min=10.0, el_max=40.0)
        cme_truth_obs = observer.compute_synthetic_obs(el_spread=0.01, cadence=5, el_min=observer.el_min,
                                                       el_max=observer.el_max)

        # Animate the truth run
        # animate_observer(model, observer, truth_key, add_flank=True, add_fov=True)

        # Save the truth CME parameters and osbervations
        truth_group.create_dataset('arrival_true', data=cme_truth.earth_arrival_time.jd)
        truth_group.create_dataset('v_true', data=cme_truth.v.value)
        truth_group.create_dataset('lon_true', data=cme_truth.longitude.to(u.deg).value)
        truth_group.create_dataset('lat_true', data=cme_truth.latitude.to(u.deg).value)
        truth_group.create_dataset('width_true', data=cme_truth.width.to(u.deg).value)
        truth_group.create_dataset('thickness_true', data=cme_truth.thickness.to(u.solRad).value)
        truth_group.create_dataset('model_flank_true', data=observer.model_flank.values)
        truth_group.create_dataset('observed_flank', data=cme_truth_obs.values)
        truth_group.create_dataset('n_members', data=n_members)
        truth_group.create_dataset('observer_lon', data=observer_lon.value)

        # Loop through the observations.
        first_pass_flag = True
        for i, row in cme_truth_obs.iterrows():

            analysis_key = "analysis_{:02d}".format(i)
            analysis_group = truth_group.create_group(analysis_key)

            t_obs = row['time']
            e_obs = row['el']

            analysis_group.create_dataset('t_obs', data=t_obs)
            analysis_group.create_dataset('e_obs', data=e_obs)

            speeds = np.zeros(n_members)
            arrivals = np.zeros(n_members)
            lons = np.zeros(n_members)
            lats = np.zeros(n_members)
            widths = np.zeros(n_members)
            thicks = np.zeros(n_members)
            likelihood = np.zeros(n_members)

            for j in range(n_members):

                # Perturb the CME, solve, and get the observer data.
                if first_pass_flag:
                    cme_ens = perturb_cone_cme(cme_base)
                else:
                    cme_ens = updated_cmes[j]

                model.solve([cme_ens])
                cme_ens = model.cmes[0]
                ens_observer = Observer(model, observer_lon, el_min=4.0, el_max=40.0)

                # Collect all the ensemble elongation profiles together. 
                if j == 0: 
                    ens_profiles = ens_observer.model_flank.copy()
                    ens_profiles.drop(columns=['r', 'lon'], inplace=True)
                    ens_profiles.rename(columns={'el': 'e_{:02d}'.format(j)}, inplace=True)
                else:
                    ens_profiles['e_{:02d}'.format(j)] = ens_observer.model_flank['el'].copy()

                # Compute the likelihood of the observation given the members profile
                profile = ens_observer.model_flank.copy()
                # Find closest time - there should be an exact match, but this is safer
                # TODO - add check that closest value isn't too far away?
                id_obs = np.argmin(np.abs(profile['time'].values - t_obs))
                e_mod = profile.loc[id_obs, 'el']
                # Use Gaussian likelihood
                likelihood[j] = st.norm.pdf(e_obs, loc=e_mod, scale=0.2)

                # Save this members CME data
                speeds[j] = cme_ens.v.value
                lons[j] = cme_ens.longitude.to(u.deg).value
                lats[j] = cme_ens.latitude.to(u.deg).value
                widths[j] = cme_ens.width.to(u.deg).value
                thicks[j] = cme_ens.thickness.to(u.solRad).value
                arrivals[j] = cme_ens.earth_arrival_time.jd

            first_pass_flag = False

            weights = likelihood / np.nansum(likelihood)

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

            out_file.flush()

            # Get resampled CMEs for next iteration
            updated_cmes = compute_resampling(speeds, lons, lats, widths, thicks, weights)

            if i == 8:
                break

    out_file.close()
    return


if __name__ == "__main__":
    run_experiment()
    #run_experiment_random_background()
