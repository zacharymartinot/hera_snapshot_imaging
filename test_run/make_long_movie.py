import numpy as np
import numba as nb

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8,6]
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.dpi'] = 100

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox

import pyuvdata
import hera_cal
import glob
import os
import copy
import yaml
import time
import multiprocessing as mp

from scipy.interpolate import interp1d

from astropy import units
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time

from pyuvdata import UVData, UVCal
from pyuvdata.utils import uvcalibrate

from hera_snapshot_imaging import HERASnapshotImager

##############
TEST_MODE = False
##############

h4c_data_dir = '/lustre/aoc/projects/hera/H4C/2459122/'

h4c_data_paths = sorted(glob.glob(h4c_data_dir + 'zen.*.sum.uvh5'))

h4c_cal_files = [h4c_data_paths[f].replace('sum.uvh5', 'sum.smooth_abs.calfits') for f in range(len(h4c_data_paths))]

a_priori_flags_path = '/lustre/aoc/projects/hera/H4C/h4c_software/hera_pipelines/pipelines/h4c/rtp/v1/stage_2_a_priori_flags/2459122.yaml'

with open(a_priori_flags_path, 'r') as f:
    flag_data = yaml.safe_load(f)
    ex_ants = flag_data['ex_ants']

uvd = pyuvdata.UVData()
uvd.read(h4c_data_paths[0], read_data=False)
h4c_good_ants = [a for a in uvd.get_ants() if a not in ex_ants]

freq_chans = [685]

def get_calibrated_h4c_file_data(df_cf):
    df, cf = df_cf
    uvd = UVData()

    uvd.read(df, antenna_nums=h4c_good_ants, freq_chans=freq_chans, polarizations=[-5,-6], run_check=False)
    uvc = UVCal()
    uvc.read_calfits(cf, run_check=False)
    uvcalibrate(uvd, uvc)

    uvd.compress_by_redundancy(method='average', tol=0.4, inplace=True)
    uvd.reorder_blts(order="time", conj_convention='ant1<ant2')

    return uvd

def get_all_times_calibrated_h4c_data(data_files, cal_files, N_threads=1):
    print("Getting data files using {} threads...".format(N_threads))
    if N_threads == 1:
        uvds = [get_calibrated_h4c_file_data(df_cf) for df_cf in zip(data_files, cal_files)]
    else:

        with mp.Pool(N_threads) as p:
            uvds = list(p.map(get_calibrated_h4c_file_data, list(zip(data_files, cal_files))))

    print("Concatenating...")
    uvd = uvds[0].fast_concat(uvds[1:], axis='blt', run_check=False)

    return uvd

print("Getting data...")
t1 = time.time()

if TEST_MODE:
    uvd = get_all_times_calibrated_h4c_data(h4c_data_paths[::95], h4c_cal_files[::95], N_threads=15)
else:
    uvd = get_all_times_calibrated_h4c_data(h4c_data_paths, h4c_cal_files, N_threads=15)

t2 = time.time()
print("Done. Time:", (t2 - t1), ' seconds.')

print("Computing images...")
t1 = time.time()
HSI = HERASnapshotImager(uvd, pad_factor=2.0, delta_uv=1.0, uv_taper_scale=0.5)
HSI.compute_images()
t2 = time.time()
print("Time:", (t2 - t1), ' seconds.')

print("Plotting...")

save_dir = '/users/zmartino/zmartino/plots/snapshot_image_tests_2022Sept11/long_movie_test/'

save_path_base = save_dir + 'integration_{}.png'
l_ax = HSI.l_ax
m_ax = HSI.m_ax
frequency_mhz = uvd.freq_array[0,0]*1e-6

def lm2altaz(l,m):
    n = np.sqrt(l**2. + m**2.)
    az = np.arctan2(l, m)
    alt = np.arccos(n)
    return alt, az

def altaz2lm(alt, az):
    l = np.cos(alt)*np.sin(az)
    m = np.cos(alt)*np.cos(az)
    return l, m

HERA_LAT = np.radians(-30.72152777777791)
HERA_LON = np.radians(21.428305555555557)
HERA_HEIGHT = 1073.0000000093132  # meters

hera_location = EarthLocation(lat=HERA_LAT*units.rad, lon=HERA_LON*units.rad, height=HERA_HEIGHT*units.meter)

def get_lines_of_constant_dec(decs_deg, jd, ra0, hera_location):
    Npts = 21
    obstime = Time(jd, format='jd', scale='ut1')
    ra_pts = np.linspace(ra0 - np.radians(30.), ra0 + np.radians(30.), Npts, endpoint=True)

    ra_pts = np.mod(ra_pts, 2*np.pi)

    dec_curves = {}
    for dec_deg in decs_deg:
        dec_pts = np.radians(dec_deg)*np.ones(Npts)

        radec_coords = SkyCoord(ra=ra_pts*units.rad, dec=dec_pts*units.rad, frame='icrs')
        altaz = radec_coords.transform_to(AltAz(obstime=obstime, location=hera_location))
        alt = altaz.alt.rad
        az = altaz.az.rad

        l_p, m_p = altaz2lm(alt, az)

        dec_curve = interp1d(l_p, m_p, kind='cubic', bounds_error=False, fill_value=np.nan)


        dec_curves[dec_deg] = dec_curve

    return dec_curves

def get_lines_of_constant_ra(ras_deg, jd, dec0, hera_location):
    Npts = 21
    obstime = Time(jd, format='jd', scale='ut1')
    dec_pts = np.linspace(dec0 - np.radians(29.9), dec0+np.radians(30.), Npts)

    ra_curves = {}
    for ra_deg in ras_deg:

        ra_pts = np.radians(ra_deg)*np.ones(Npts)

        radec_coords = SkyCoord(ra=ra_pts*units.rad, dec=dec_pts*units.rad, frame='icrs')
        altaz = radec_coords.transform_to(AltAz(obstime=obstime, location=hera_location))
        alt = altaz.alt.rad
        az = altaz.az.rad

        l_p, m_p = altaz2lm(alt, az)

        ra_curve = interp1d(l_p, m_p, kind='cubic', bounds_error=False, fill_value=np.nan)

        ra_curves[ra_deg] = ra_curve

    return ra_curves

@nb.vectorize("float64(float64, float64)")
def ra_distance_degrees(ra1, ra2):
    abs_diff = np.abs(ra1 - ra2)
    return min(abs_diff, abs(abs_diff - 360.))

decs = np.linspace(-20,-40,5)
dec_curves = get_lines_of_constant_dec(decs, HSI.times[0], HSI.lsts[0], hera_location)

all_ras = np.linspace(0, 360, 360//5 + 1)
dec0 = HERA_LAT

ra0 = np.degrees(HSI.lsts[0])

ras = all_ras[np.where(ra_distance_degrees(all_ras, ra0) < 25.)]

ra_curves = get_lines_of_constant_ra(ras, HSI.times[0], dec0, hera_location)

print('making test plots')

fig, ax = plt.subplots(1, 1)

dec_tick_locs = []
for dec in dec_curves:
    dec_curve = dec_curves[dec]

    ax.plot(l_ax, dec_curve(l_ax), '--k', linewidth=1)
    dec_tick_locs.append(dec_curve(-0.2))

dec_tick_labels = list(map(int,decs))

ra_tick_locs = []
for ra in ra_curves:
    ra_curve = ra_curves[ra]
    m_vals = ra_curve(-l_ax)
    ax.plot(l_ax, m_vals, '--k', linewidth=1)
    l_idx = np.nanargmin(abs(-0.2 - m_vals))

    ra_tick_locs.append(l_ax[l_idx])

ra_tick_labels = list(map(int, ras))

ax.set_xticks(ra_tick_locs)
ax.set_xticklabels(ra_tick_labels)

ax.set_yticks(dec_tick_locs)
ax.set_yticklabels(dec_tick_labels)

edge = 0.2
ax.set_xlim(-edge,edge)
ax.set_ylim(-edge, edge)

ax.set_xlabel('Right Ascension (degrees)')
ax.set_ylabel('Declination (degrees)')
fig.savefig(save_dir + 'coordinate_lines.png')
plt.close()

extent = [l_ax[0], l_ax[-1], m_ax[-1], m_ax[0]]
bbox_inches = None
for tidx in range(0,HSI.Nt):

    fig, ax = plt.subplots(1,1,figsize=(8,8),facecolor='white')
    img_est = HSI.img_ests[tidx]
    img = ax.imshow(img_est, extent=extent, aspect='equal', cmap='Greys_r')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    cbar = fig.colorbar(img, cax=cax)

    dec_tick_locs = []
    for dec in dec_curves:
        dec_curve = dec_curves[dec]

        ax.plot(l_ax, dec_curve(l_ax), '--k', linewidth=1)
        dec_tick_locs.append(dec_curve(-0.2))

    dec_tick_labels = list(map(int,decs))

    ra0 = np.degrees(HSI.lsts[tidx])

    ras = all_ras[np.where(ra_distance_degrees(all_ras, ra0) < 25.)]

    ra_curves = get_lines_of_constant_ra(ras, HSI.times[tidx], dec0, hera_location)

    ra_tick_locs = []
    for ra in ra_curves:
        ra_curve = ra_curves[ra]
        try:
            m_vals = ra_curve(-l_ax)
            l_idx = np.nanargmin(abs(-0.2 - m_vals))

            ax.plot(l_ax, m_vals, '--k', linewidth=1)

            ra_tick_locs.append(l_ax[l_idx])
        except ValueError:
            print(ra, ra0)
            ax.plot([0., 0.], [-1,1], '--k', linewidth=1)

            ra_tick_locs.append(0.0)

    ra_tick_labels = list(map(int, ras))

    ax.set_xticks(ra_tick_locs)
    ax.set_xticklabels(ra_tick_labels)

    ax.set_yticks(dec_tick_locs)
    ax.set_yticklabels(dec_tick_labels)

    edge = 0.2

    ax.set_xlim(-edge, edge)
    ax.set_ylim(-edge, edge)

    ax.set_xlabel('Right Ascension (degrees)')
    ax.set_ylabel('Declination (degrees)')

    ax.set_title('JD 2459122, Frequency ' + str(np.around(frequency_mhz, 2)) + ' MHz, (wrong/unnormalized magnitude)')

    lst = np.around(12/np.pi * HSI.lsts[tidx], 2)
    ax.text(0.05, 0.95, f'LST {lst} hr', fontsize=20, color='white', transform=ax.transAxes)

    if bbox_inches is None:
        tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())

        x0 = tight_bbox._bbox.x0/fig.dpi - 0.25
        y0 = tight_bbox._bbox.y0/fig.dpi - 0.25
        x1 = fig.get_figwidth() - 0.01
        y1 = tight_bbox._bbox.y1/fig.dpi + 0.25
        bbox_inches = Bbox([[x0,y0],[x1,y1]])

    # some bug make this necessary. See https://github.com/matplotlib/matplotlib/issues/22625
    save_path_name = save_path_base.format(str(tidx).rjust(4,'0'))
    fig.savefig(save_path_name, bbox_inches=None);
    fig.savefig(save_path_name, bbox_inches=bbox_inches);
    plt.close()

print("Done with plots.")
# print("Building movie...")
# 
# os.system("~/zmartino/ffmpeg/ffmpeg-git-20220910-amd64-static/ffmpeg -framerate 48 -pattern_type glob -i 'integration_*.png' test_movie.webm")

print("Done")
