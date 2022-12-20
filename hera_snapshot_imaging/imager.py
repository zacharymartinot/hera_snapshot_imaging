import numpy as np
import numba as nb

from scipy import linalg
from scipy.spatial.distance import cdist

import glob
import os
import copy
import yaml

import pyuvdata
import hera_cal

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

class HERASnapshotImager:

    def __init__(self, uvd, pad_factor=3.0, delta_uv=0.5, uv_taper_scale=None, output_crop_lim=None):

        # uvd.compress_by_redundancy(method='average', tol=0.4, inplace=True)
        # uvd.reorder_blts(order="time", conj_convention='ant1<ant2')
        # uvd.select(bls=[(i,j) for (i,j) in uvd.get_antpairs() if i != j])


        self.uvd = uvd
        self.uv_taper_scale = uv_taper_scale
        self.output_crop_lim = output_crop_lim

        self.times = uvd.time_array[::uvd.Nbls]
        self.lsts = uvd.lst_array[::uvd.Nbls]
        self.nu_hz = uvd.freq_array[0]

        self.Nt = len(self.lsts)
        self.Nf = len(self.nu_hz)

        self.ant_pos, self.ant_num = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        self.ant_map = {a:r for (a,r) in zip(self.ant_num, self.ant_pos)}

        bl_nums = uvd.get_baseline_nums()

        # set of baselines to be used
        # self.b_vectors = np.array([
        #     self.ant_map[j] - self.ant_map[i] for (i,j) in list(map(uvdr.baseline_to_antnums, bl_nums))]
        # ])
        self.b_vectors = uvd.uvw_array[0:uvd.Nbls]

        # include the reflections of each baseline
        self.b_vectors = np.r_[self.b_vectors, -self.b_vectors]

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

    def compute_images(self, average_frequencies=True, compute_psf=False):

        if average_frequencies:
            self.img_ests = np.zeros((self.Nt, self.Np, self.Np))

        else:
            self.img_ests = np.zeros((self.Nf, self.Nt, self.Np, self.Np))

        if compute_psf:
            self.psf_ests = np.zeros_like(self.img_ests)

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
                uv_taper = np.ones_like(self.uv_grid)

            for i_t in range(self.Nt):

                blt_slice = slice(i_t*self.uvd.Nbls, (i_t+1)*self.uvd.Nbls)
                uv_vals = 0.5*(self.uvd.data_array[blt_slice,0,i_f,0] + self.uvd.data_array[blt_slice,0,i_f,1])

                uv_vals = np.r_[uv_vals, np.conj(uv_vals)]
                K.compute_coefficients(uv_vals)
                uv_interp = uv_taper * K(self.uv_grid)

                if compute_psf:
                    K.compute_coefficients(np.ones_like(uv_vals))
                    uv_samp = uv_taper * K(self.uv_grid)


                img_est = np.fft.fft2(np.fft.ifftshift(uv_interp, axes=(0,1)))
                img_est = np.fft.fftshift(img_est, axes=(0,1))
                img_est = np.real(img_est)
                img_est = np.fliplr(img_est)

                img_est *= (2*np.pi)**2. * self.delta_uv**2. / img_est.size

                if compute_psf:
                    psf_est = np.fft.fft2(np.fft.ifftshift(uv_samp, axes=(0,1)))
                    psf_est = np.fft.fftshift(psf_est, axes=(0,1))
                    psf_est = np.real(psf_est)
                    psf_est = np.fliplr(psf_est)

                    psf_est *= (2*np.pi)**2. * self.delta_uv**2. / img_est.size

                if average_frequencies:
                    self.img_ests[i_t] += img_est

                    if compute_psf:
                        self.psf_ests[i_t] += psf_est

                else:
                    self.img_ests[i_f, i_t] = img_est

                    if compute_psf:
                        self.psf_ests[i_f, i_t] = psf_est

        if average_frequencies:
            self.img_ests /= self.Nf

            if compute_psf:

                self.psf_ests /= self.Nf

    def print_image_array_layout(self):
        if self.img_ests is None:
            print("No images computed yet.")
        elif len(self.img_ests) == 3:
            print("Axes:(Times, l, m)")
        elif len(self.img_ests) == 4:
            print("Axis:(Frequencies, Times, l, m)")
        else:
            print("WTF mate?")

def recommended_preprocessing(uvd, uvc):

    return
