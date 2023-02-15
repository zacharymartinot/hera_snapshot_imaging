import numpy as np
import numba as nb

from scipy import linalg
from scipy.spatial.distance import cdist

from scipy.spatial import ConvexHull
from scipy.interpolate import CloughTocher2DInterpolator, PchipInterpolator
from scipy.interpolate import interp1d

from astropy import units
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox

c_mps = 299792458.0 # speed of light in meters per second

@nb.njit
def gaussian(r, eps):
    return np.exp(-(r/eps)**2)

@nb.njit("c16[:,:](f8[:,:,:], f8[:,:], f8[:], c16[:])", fastmath=True, parallel=True, cache=True)
def eval_uv_interp(uv_grid, uv_nodes, eps_arr, coeffs):
    tol = 1e-14

    Nu, Nv = uv_grid.shape[0], uv_grid.shape[1]
    Nn = uv_nodes.shape[0]

    u_ax = uv_grid[0,:,0]
    v_ax = uv_grid[:,0,1]

    delta_uv = u_ax[1] - u_ax[0]

    uv_interp = np.zeros((Nu, Nv), dtype=nb.complex128)

    for nn in nb.prange(Nn):
        eps_arr[nn] = eps_arr[nn]
        uv_n = uv_nodes[nn]
        c_n = coeffs[nn]

        # half size of the bounding square
        kernel_size = eps_arr[nn] * np.sqrt(np.log(1/tol))

        # half the number of grid points on a side
        Nk = int(np.ceil(kernel_size / delta_uv))

        # indices of nearest grid point to the node
        ic_un = np.argmin(np.abs(uv_n[0] - u_ax))
        ic_vn = np.argmin(np.abs(uv_n[1] - v_ax))

        # indices of the bounding square on the grid
        i0_u = ic_un - Nk
        i1_u = ic_un + Nk + 1

        i0_v = ic_vn - Nk
        i1_v = ic_vn + Nk + 1

        for ii in range(i0_v, i1_v):
            for jj in range(i0_u, i1_u):

                r_eval = np.sqrt(np.sum(np.square(uv_grid[ii,jj,:] - uv_nodes[nn,:])))

                if r_eval < kernel_size:
                    uv_interp[ii,jj] += gaussian(r_eval, eps_arr[nn]) * coeffs[nn]

    return uv_interp

class Interpolator:

    def __init__(self, uv_nodes, eps, uv_node_diffs=None, uv_vals=None):
        self.uv_nodes = uv_nodes
        self.eps = eps

        if uv_node_diffs is None:
            self.uv_node_diffs = cdist(uv_node_diffs, uv_node_diffs)
        else:
            self.uv_node_diffs = uv_node_diffs

        self.S_mat = gaussian(self.uv_node_diffs, self.eps)

        self.lu_and_piv = linalg.lu_factor(self.S_mat)

        if uv_vals is not None:
            self.uv_vals = uv_vals
            self.compute_coefficients(uv_vals)

    def compute_coefficients(self, uv_vals):
        self.coeffs = linalg.lu_solve(self.lu_and_piv, uv_vals)

    def __call__(self, uv_pts):
        uv_interp = eval_uv_interp(uv_pts, self.uv_nodes, self.eps, self.coeffs)

        return uv_interp

def get_bounding_curve(uv_vectors):
    uv_convex_hull = ConvexHull(uv_vectors)
    chv = uv_vectors[uv_convex_hull.vertices]
    chv_shift = np.concatenate((chv[1:], chv[0].reshape(1,-1)), axis=0)
    edge_points = 0.5*(chv_shift - chv) + chv

    t_pts = np.unwrap(np.arctan2(edge_points[:,1], edge_points[:,0]))
    t_pts = np.r_[t_pts, t_pts[0] + 2*np.pi]
    t0 = t_pts[0]

    edge_points_periodic = np.r_[edge_points, edge_points[0].reshape(1,-1)]

    bounding_curve = PchipInterpolator(t_pts, edge_points_periodic)

    return bounding_curve, t0

def grid_boundary_angle(uv_grid, t0):
    uv_t = np.arctan2(uv_grid[...,1], uv_grid[...,0])
    uv_t[np.where(uv_t < t0)] += 2*np.pi
    return uv_t

def get_taper(uv_grid, uv_vectors, taper_type='hann'):

    bounding_curve, t0 = get_bounding_curve(uv_vectors)

    uv_grid_angles = grid_boundary_angle(uv_grid, t0)

    uv_grid_boundary_points = bounding_curve(uv_grid_angles)

    L_grid = np.sqrt(np.sum(np.square(uv_grid_boundary_points), axis=-1))
    uv_grid_length = np.sqrt(np.sum(np.square(uv_grid), axis=-1))

    interior_pts = np.asarray(uv_grid_length < L_grid)
    taper = np.zeros(uv_grid.shape[:2])
    if taper_type == 'hann':
        x = uv_grid_length[interior_pts]
        L = L_grid[interior_pts]
        taper[interior_pts] = np.cos(0.5*np.pi * x / L)**2
    elif taper_type == 'gaussian':
        r = uv_grid_length[interior_pts]
        eps = np.max(L_grid)/np.sqrt(np.log(1e8))
        taper[interior_pts] = np.exp(-(r/eps)**2)

    else:
        taper[interior_pts] = 1.0

    return taper

class HERASnapshotImager:

    def __init__(self, vis_data, b_vectors, freqs_hz, lsts, times,
                 pad_factor=2.0, delta_uv=0.5):

        self.times = times
        self.lsts = lsts
        self.nu_hz = freqs_hz

        self.Nt = len(self.lsts)
        self.Nf = len(self.nu_hz)
        self.Nb = b_vectors.shape[0]

        self.b_vectors = b_vectors
        # include the reflections of each baseline
        self.b_vectors = np.r_[self.b_vectors, -self.b_vectors]

        self.vis_data = vis_data

        self.max_be = np.max(abs(self.b_vectors[:,0]))
        self.max_bn = np.max(abs(self.b_vectors[:,1]))

        self.b_diff_mat = cdist(self.b_vectors, self.b_vectors)
        self.b_min_dist = ( self.b_diff_mat + np.diag([self.b_diff_mat.max()]*self.b_diff_mat.shape[0]) ).min(axis=1)

        max_u = self.max_be * np.max(self.nu_hz / c_mps)
        max_v = self.max_bn * np.max(self.nu_hz / c_mps)
        max_uv = max(max_u, max_v)

        self.pad_factor = pad_factor
        self.Nuv = int(np.ceil(self.pad_factor*max(max_u,max_v)/0.5))

        self.delta_uv = delta_uv
        self.u_ax = self.delta_uv*(np.arange(-self.Nuv, self.Nuv+1))
        self.v_ax = self.delta_uv*(np.arange(-self.Nuv, self.Nuv+1))

        self.Np = 2*self.Nuv + 1 # image points per side
        uu, vv = np.meshgrid(self.u_ax, self.v_ax, indexing='xy')

        self.uv_grid = np.zeros(uu.shape + (3,))
        self.uv_grid[:,:,0] = uu
        self.uv_grid[:,:,1] = vv

        l_ax = np.fft.fftfreq(self.u_ax.size, d=np.diff(self.u_ax)[0])
        l_ax = np.fft.fftshift(l_ax)

        self.l_ax = l_ax
        self.m_ax = np.copy(l_ax)

    def compute_gridded_mfs_images(self, taper_type='hann', gaussian_taper_tol=1e-8, kernel_size=0.25, weighted=True):

        self.gaussian_taper_tol = gaussian_taper_tol

        self.mfs_images = np.zeros((self.Nt, self.Np, self.Np))

        inv_lambda = self.nu_hz / c_mps
        uv_vectors = np.concatenate([nc * self.b_vectors for nc in inv_lambda], axis=0)

        if taper_type == 'gaussian':

            uv_taper_scale = np.sqrt(np.log(1/gaussian_taper_tol)) * (np.max(self.nu_hz)/ c_mps) * max(self.max_be, self.max_bn)

            uv_taper = np.exp(-(self.uv_grid[...,0]**2. + self.uv_grid[...,1]**2.)/uv_taper_scale**2.)

        else:
            # hann or top-hat over convex hull of uv samples
            uv_taper = get_taper(self.uv_grid[...,:2], uv_vectors[:,:2], taper_type=taper_type)

        eps = kernel_size*(np.mean(self.nu_hz) / c_mps) * np.min(self.b_min_dist)
        eps = eps * np.ones(uv_vectors.shape[0])

        if weighted:

            unit_data = np.ones(uv_vectors.shape[0], dtype=complex)

            sampling = np.real(eval_uv_interp(self.uv_grid, uv_vectors, eps, unit_data))
            r = 1e-2
            self.weights = 1 / (r + sampling)

            uv_taper *= self.weights

        self.uv_taper = uv_taper
        for i_t in range(self.Nt):

            uv_vals = np.concatenate([np.r_[self.vis_data[i_t, i_f], np.conj(self.vis_data[i_t, i_f])] for i_f in range(self.Nf)],axis=0)

            uv_interp = eval_uv_interp(self.uv_grid, uv_vectors, eps, uv_vals)

            uv_interp *= uv_taper

            mfs_img = np.fft.fft2((np.fft.ifftshift(uv_interp, axes=(0,1))))
            mfs_img = np.fft.fftshift(mfs_img, axes=(0,1))
            mfs_img = np.real(mfs_img) / ( 2*np.sum(uv_taper)*self.delta_uv**2. )
            mfs_img = np.fliplr(mfs_img)

            self.mfs_images[i_t] = mfs_img

    def compute_single_channel_spline_images(self, taper_type='hann'):

        self.img_ests = np.zeros((self.Nf, self.Nt, self.Np, self.Np))

        for i_f in range(self.Nf):

            nu = self.nu_hz[i_f]
            uv_vectors = (nu / c_mps) * self.b_vectors[:,:2]
            uv_taper = get_taper(self.uv_grid, uv_vectors, taper_type=taper_type)

            for i_t in range(self.Nt):
                uv_vals = np.r_[self.vis_data[i_t, i_f], np.conj(self.vis_data[i_t, i_f])]

                intp_spline = CloughTocher2DInterpolator(uv_vectors, uv_vals, fill_value=0.0)
                uv_interp = uv_taper * intp_spline(self.uv_grid[...,:2])
                img_est = np.fft.fft2(np.fft.ifftshift(uv_interp, axes=(0,1)))
                img_est = np.fft.fftshift(img_est, axes=(0,1))
                img_est = np.real(img_est)
                img_est = np.fliplr(img_est)

                img_est *= (2*np.pi)**2. * self.delta_uv**2. / img_est.size

                self.img_ests[i_f, i_t] = img_est

    def compute_single_channel_rbf_images(self, uv_taper_scale=0.5):

        self.uv_taper_scale = uv_taper_scale

        self.img_ests = np.zeros((self.Nf, self.Nt, self.Np, self.Np))

        for i_f in range(self.Nf):

            nu = self.nu_hz[i_f]
            uv_locs = (nu / c_mps) * self.b_vectors
            eps = (nu / c_mps) * self.b_min_dist

            uv_locs_diff = (nu / c_mps) * self.b_diff_mat

            K = Interpolator(uv_locs, eps, uv_node_diffs=uv_locs_diff)

            if self.uv_taper_scale is not None:
                uv_taper_scale = (nu / c_mps) * max(self.max_be, self.max_bn) * self.uv_taper_scale

                uv_taper = np.exp(-(self.uv_grid[...,0]**2. + self.uv_grid[...,1]**2.)/uv_taper_scale**2.)
            else:
                uv_taper = np.ones_like(self.uv_grid[:,:,0])

            for i_t in range(self.Nt):

                uv_vals = np.r_[self.vis_data[i_t, i_f], np.conj(self.vis_data[i_t, i_f])]

                K.compute_coefficients(uv_vals)
                uv_interp = uv_taper * K(self.uv_grid)

                img_est = np.fft.fft2(np.fft.ifftshift(uv_interp, axes=(0,1)))
                img_est = np.fft.fftshift(img_est, axes=(0,1))
                img_est = np.real(img_est)
                img_est = np.fliplr(img_est)

                img_est *= (2*np.pi)**2. * self.delta_uv**2. / img_est.size

                self.img_ests[i_f, i_t] = img_est

    def compute_spline_mfs_images(self, taper_type='hann', doing_it_anyway=False):

        if not doing_it_anyway:
            raise Exception("This doesn't work, don't do this.")

        self.mfs_images = np.zeros((self.Nt, self.Np, self.Np))

        inv_lambda = self.nu_hz / c_mps
        uv_vectors = np.concatenate([nc * self.b_vectors[:,:2] for nc in inv_lambda], axis=0)

        uv_taper = get_taper(self.uv_grid, uv_vectors, taper_type=taper_type)

        for i_t in range(self.Nt):

            uv_vals = np.concatenate([np.r_[self.vis_data[i_t, i_f], np.conj(self.vis_data[i_t, i_f])] for i_f in range(self.Nf)],axis=0)

            intp_spline = CloughTocher2DInterpolator(uv_vectors, uv_vals, fill_value=0.0)

            uv_interp = uv_taper * intp_spline(self.uv_grid[...,:2])

            mfs_img = np.fft.fft2((np.fft.ifftshift(uv_interp, axes=(0,1))))
            mfs_img = np.fft.fftshift(mfs_img, axes=(0,1))
            mfs_img = np.real(mfs_img) * (2*np.pi)**2 * self.delta_uv**2 / mfs_img.size
            mfs_img = np.fliplr(mfs_img)

            self.mfs_images[i_t] = mfs_img

@nb.vectorize("float64(float64, float64)")
def ra_distance_degrees(ra1, ra2):
    abs_diff = np.abs(ra1 - ra2)
    return min(abs_diff, abs(abs_diff - 360.))

def image_plot(HSI, tidx, fidx, vmin=None,vmax=None, mfs=False, edge=0.15):

    l_ax = HSI.l_ax
    m_ax = HSI.m_ax
    frequency_mhz = HSI.nu_hz[fidx]*1e-6

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

    decs = np.linspace(-20,-40,5)
    dec_curves = get_lines_of_constant_dec(decs, HSI.times[tidx], HSI.lsts[tidx], hera_location)

    all_ras = np.linspace(0, 360, 360//5 + 1)
    dec0 = HERA_LAT

    ra0 = np.degrees(HSI.lsts[tidx])

    ras = all_ras[np.where(ra_distance_degrees(all_ras, ra0) < 25.)]

    ra_curves = get_lines_of_constant_ra(ras, HSI.times[tidx], dec0, hera_location)

    dec_tick_locs = []
    for dec in dec_curves:
        dec_curve = dec_curves[dec]

        dec_tick_locs.append(dec_curve(-0.2))

    dec_tick_labels = list(map(int,decs))

    ra_tick_locs = []
    for ra in ra_curves:
        ra_curve = ra_curves[ra]
        m_vals = ra_curve(-l_ax)
        l_idx = np.nanargmin(abs(-0.2 - m_vals))

        ra_tick_locs.append(l_ax[l_idx])

    ra_tick_labels = list(map(int, ras))

    extent = [l_ax[0], l_ax[-1], m_ax[-1], m_ax[0]]

    fig, ax = plt.subplots(1,1,figsize=(8,8),facecolor='white')
    if mfs:
        img_est = HSI.mfs_images[tidx]
    else:
        img_est = HSI.img_ests[fidx, tidx]

    img = ax.imshow(img_est, extent=extent, aspect='equal', cmap='Greys_r', vmin=vmin, vmax=vmax)

    ax.plot(0., 0., 'or', ms=3, markerfacecolor='None')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    cbar = fig.colorbar(img, cax=cax)

    dec_tick_locs = []
    for dec in dec_curves:
        dec_curve = dec_curves[dec]

        ax.plot(l_ax, dec_curve(l_ax), '--w', linewidth=1)
        dec_tick_locs.append(dec_curve(-0.2))

    dec_tick_labels = list(map(int,decs))

    ra_tick_locs = []
    for ra in ra_curves:
        ra_curve = ra_curves[ra]
        try:
            m_vals = ra_curve(-l_ax)
            l_idx = np.nanargmin(abs(-0.2 - m_vals))

            ax.plot(l_ax, m_vals, '--w', linewidth=1)

            ra_tick_locs.append(l_ax[l_idx])
        except ValueError:
            print(ra, ra0)
            ax.plot([0., 0.], [-1,1], '--w', linewidth=1)

            ra_tick_locs.append(0.0)

    ra_tick_labels = list(map(int, ras))

    ax.set_xticks(ra_tick_locs)
    ax.set_xticklabels(ra_tick_labels)

    ax.set_yticks(dec_tick_locs)
    ax.set_yticklabels(dec_tick_labels)

#     edge = 0.15

    ax.set_xlim(-edge, edge)
    ax.set_ylim(-edge, edge)

    ax.set_xlabel('Right Ascension (degrees)')
    ax.set_ylabel('Declination (degrees)')

    ax.set_title(f'JD {int(HSI.times[tidx]) }, Frequency ' + str(np.around(frequency_mhz, 2)) + ' MHz')

    lst = np.around(12/np.pi * HSI.lsts[tidx], 2)
    ax.text(0.05, 0.95, f'LST {lst} hr', fontsize=20, color='white', transform=ax.transAxes)

    plt.show()
