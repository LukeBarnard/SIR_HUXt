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
    v_sub = model.v_grid.value[id_t, :, :].copy()
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
    cme_colors = ['darkorange', 'c', 'm', 'y', 'deeppink', 'r']
    for j, cme in enumerate(model.cmes):
        cid = np.mod(j, len(cme_colors))
        cme_lons = cme.coords[id_t]['lon']
        cme_r = cme.coords[id_t]['r'].to(u.solRad)
        if np.any(np.isfinite(cme_r)):
            # Pad out to close the profile.
            cme_lons = np.append(cme_lons, cme_lons[0])
            cme_r = np.append(cme_r, cme_r[0])
            ax.plot(cme_lons, cme_r, '-', color=cme_colors[cid], linewidth=3, zorder=3)
            
    #cme = model.cmes[0]
    #ax.plot(cme.coords[id_t]['lon'], cme.coords[id_t]['r'], '-', color='darkorange', linewidth=3, zorder=3)
    
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
