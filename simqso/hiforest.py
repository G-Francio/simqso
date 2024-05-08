#!/usr/bin/env python

from __future__ import print_function

import multiprocessing
import os
from collections import namedtuple
from functools import partial

import numpy as np
import scipy.constants as const
from astropy.io import fits
from astropy.table import Table, vstack
from numba import jit
from scipy.stats import poisson
from tqdm import tqdm

from .sqbase import datadir, fixed_R_dispersion, resample

# shorthands
exp, sqrt, log = np.exp, np.sqrt, np.log
c_kms = const.c / 1e3
sqrt_pi = sqrt(np.pi)
sigma_c = 6.33e-18  # cm^-2
fourpi = 4 * np.pi


def _getlinelistdata():
    # Line list obtained from Prochaska's XIDL code
    # https://svn.ucolick.org/xidl/trunk/Spec/Lines/all_lin.fits
    linelist = fits.getdata(os.path.join(datadir, "all_lin.fits"))
    Hlines = np.array([i for i in range(linelist.size) if "HI" in linelist.ION[i]])
    # if b'HI' in linelist.ION[i]])
    transitionParams = {}
    for n, idx in enumerate(Hlines[::-1], start=2):
        transitionParams[n] = (
            linelist.WREST[idx],
            linelist.F[idx],
            linelist.GAMMA[idx],
        )
    return transitionParams


transitionParams = _getlinelistdata()

# default is to go up to 32->1
default_lymanseries_range = (2, 33)


def generate_los(model, z_min, z_max, rng):
    """
    Given a model for the distribution of absorption systems, generate
    a random line-of-sight populated with absorbers.
    returns (z, logNHI, b) for each absorption system.
    """
    abs_dtype = [("z", np.float32), ("logNHI", np.float32), ("b", np.float32)]
    absorbers = []
    for _, p in model.items():
        if z_min > p["zrange"][1] or z_max < p["zrange"][0]:
            # outside the redshift range of this forest component
            continue
        # parameters for the forest component (LLS, etc.) absorber distribution
        NHI_min, NHI_max = p["logNHrange"]
        NHI_min, NHI_max = 10**NHI_min, 10**NHI_max
        z1 = max(z_min, p["zrange"][0])
        z2 = min(z_max, p["zrange"][1])
        beta = p["beta"]

        # shorthands
        m_beta_p_1 = -beta + 1
        gamma_p_1 = p["gamma"] + 1

        # The following is just a lot of inverting distributions

        # Expectation for the number of absorbers at this redshift
        #  (inverting n(z) = N0*(1+z)^gamma)
        N = (p["N0"] / gamma_p_1) * ((1 + z2) ** gamma_p_1 - (1 + z1) ** gamma_p_1)
        # sample from a Poisson distribution for <N>
        n = poisson.rvs(N, size=1)[0]

        # 1 - Invert the dN/dz CDF to get the sample redshifts
        x = rng.random(n)  # these are just uniform [0, 1) random numbers
        z = (1 + z1) * ((((1 + z2) / (1 + z1)) ** gamma_p_1 - 1) * x + 1) ** (
            1 / gamma_p_1
        ) - 1

        # 2 - Invert the NHI CDF to get the sample column densities
        x = rng.random(n)
        NHI = NHI_min * (1 + x * ((NHI_max / NHI_min) ** m_beta_p_1 - 1)) ** (
            1 / m_beta_p_1
        )

        # 3 - Invert the b CDF to get the sample column densities OR
        #  decide to take b as a constant and make a simplified model
        try:
            b = np.array([p["b"]] * n, dtype=np.float32)
        except KeyError:
            # dn/db ~ b^-5 exp(-(b/bsig)^-4) (Hui & Rutledge 1999)
            b_sig = p["bsig"]
            b_min, b_max = p["brange"]
            x = rng.random(n)
            b = b_sig * (
                -np.log(
                    (bexp(b_max, b_sig) - bexp(b_min, b_sig)) * x + bexp(b_min, b_sig)
                )
            ) ** (-1.0 / 4)

        absorber = np.empty(n, dtype=abs_dtype)
        absorber["z"] = z
        absorber["logNHI"] = np.log10(NHI)
        absorber["b"] = b
        absorbers.append(absorber)
    absorbers = np.concatenate(absorbers)

    # return sorted by redshift
    return absorbers[absorbers["z"].argsort()]


@jit(nopython=True)
def bexp(b, b_sig):
    return np.exp(-((b / b_sig) ** -4))


@jit(nopython=True)
def voigt(a, x):
    """Tepper-Garcia 2006, footnote 4 (see erratum)"""
    x2 = x * x
    Q = 1.5 / x2
    H0 = np.exp(-x2)
    return H0 - (a / sqrt_pi) / x2 * (H0 * H0 * (4 * x2 * x2 + 7 * x2 + 4 + Q) - Q - 1)


@jit(nopython=True)
def sum_of_voigts(wave, tau_lam, c_voigt, a, lambda_z, b, tau_min, tau_max):
    """
    Given arrays of parameters, compute the summed optical depth
    spectrum of absorbers using Voigt profiles.
    Uses the Tepper-Garcia 2006 approximation for the Voigt function.
    """
    u_max = np.clip(np.sqrt(c_voigt * (a / sqrt_pi) / tau_min), 5.0, np.inf)

    # ***assumes constant velocity bin spacings***
    dv = (wave[1] - wave[0]) / (0.5 * (wave[0] + wave[1])) * c_kms
    du = dv / b
    b_norm = b / c_kms
    n_pix = (u_max / du).astype(np.int32)

    for i in range(len(a)):
        w0 = np.searchsorted(wave, lambda_z[i])
        i1 = max(0, w0 - n_pix[i])
        i2 = min(len(wave), w0 + n_pix[i])
        if np.all(tau_lam[i1:i2] > tau_max):
            continue

        # the clip is to prevent division by zero errors
        u = np.abs((wave[i1:i2] / lambda_z[i] - 1) / b_norm[i]).clip(1e-5, np.inf)
        tau_lam[i1:i2] += c_voigt[i] * voigt(a[i], u)

    return tau_lam


@jit(nopython=True)
def sum_of_continuum_absorption(wave, tau_lam, NHI, zp1, tau_min, tau_max):
    """
    Compute the summed optical depth for Lyman continuum blanketing
    given a series of absorbers with column densities NHI and
    redshifts zp1 (=1+z).
    """
    tau_c_lim = sigma_c * NHI
    lambda_z_c = 912.0 * zp1
    ii = np.where((lambda_z_c > wave[0]) & (tau_c_lim > tau_min))[0]

    # sort by decreasing column density to start with highest tau systems
    ii = ii[NHI[ii].argsort()[::-1]]

    # ending pixel (wavelength at onset of continuum absorption)
    i_end = np.searchsorted(wave, lambda_z_c[ii], side="right")

    # starting pixel - wavelength where tau drops below tauMin
    wave_start = (tau_min / tau_c_lim[ii]) ** 0.333 * wave[i_end]
    i_start = np.searchsorted(wave, wave_start)

    # now do the sum
    for i, i1, i2 in zip(ii, i_start, i_end):
        # ... only if pixels aren't already saturated
        if np.any(tau_lam[i1:i2] < tau_max):
            l1l0 = wave[i1:i2] / lambda_z_c[i]
            tau_lam[i1:i2] += tau_c_lim[i] * l1l0 * l1l0 * l1l0
    return tau_lam


# from http://stackoverflow.com/questions/42558/python-and-the-singleton-pattern
class Singleton:
    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self, *args, **kwargs):
        try:
            inst = self._instance
            # self._argcheck(*args)
        except AttributeError:
            self._instance = self._decorated(*args, **kwargs)
            inst = self._instance
        return inst

    def __call__(self):
        raise TypeError('Must be accessed through "Instance()".')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)

    # def _argcheck(self,*args):
    #    raise NotImplementedError


@Singleton
class VoigtTable(object):
    """
    Lookup table of Voigt profiles use to precompute low-density absorbers.
    """

    def __init__(self, *args, **kwargs):
        self._init_table(*args, **kwargs)

    def _argcheck(self, *args):
        assert self.dv == args[0]

    def _init_table(self, *args, **kwargs):
        (wave,) = args
        # ***assumes constant velocity bin spacings***
        dv = (wave[1] - wave[0]) / (0.5 * (wave[0] + wave[1])) * c_kms
        self.wave0 = wave[0]
        self.npix = len(wave)
        self.dv = dv
        self.dv_c = dv / c_kms
        #
        na = kwargs.get("fastvoigt_na", 20)
        loga_min = kwargs.get("fastvoigt_logamin", -8.5)
        loga_max = kwargs.get("fastvoigt_logamax", -3.0)
        gamma = kwargs.get("fastvoigt_gamma", 1.5)
        nb = kwargs.get("fastvoigt_nb", 20)
        u_range = kwargs.get("fastvoigt_urange", 10)
        # define the bins in Voigt a parameter using exponential spacings
        alpha = (loga_max - loga_min) / na**gamma
        self.logabins = np.array([loga_max - alpha * n**gamma for n in range(na)])
        # define the bins in b
        self.bbins = np.linspace(10.0, 100.0, nb)
        #
        self.xv = {}
        for j, b in enumerate(self.bbins):
            # offset slightly to avoid division by zero error
            self.xv[j] = np.arange(1e-5, u_range, dv / b)
        self.dx = np.array([len(self.xv[j]) - 1 for j in range(len(self.bbins))])
        self.voigt_tab = {}
        for i in range(na):
            self.voigt_tab[i] = {}
            for j in range(nb):
                vprof = voigt(10 ** self.logabins[i], self.xv[j])
                self.voigt_tab[i][j] = np.concatenate([vprof[::-1][1:], vprof])

    def sum_of_voigts(self, a, b, wave, c_voigt, tau_lam):
        ii = np.argmin(
            np.abs(np.log10(a)[:, np.newaxis] - self.logabins[np.newaxis, :]), axis=1
        )
        jj = np.argmin(np.abs(b[:, np.newaxis] - self.bbins[np.newaxis, :]), axis=1)
        wc = np.round((np.log(wave) - np.log(self.wave0)) / self.dv_c)
        wc = wc.astype(np.int32)
        dx = self.dx[jj]
        w1, w2 = wc - dx, wc + dx + 1
        x1, x2 = np.zeros_like(dx), 2 * dx + 1
        # off left edge of spectrum
        ll = np.where(w1 < 0)[0]
        x1[ll] = -w1[ll]
        w1[ll] = 0
        # off right edge of spectrum
        ll = np.where(w2 > self.npix)[0]
        x2[ll] = self.npix - w1[ll]
        w2[ll] = self.npix
        # within the spectrum!
        ll = np.where(~((w2 < 0) | (w1 >= self.npix) | (w2 - w1 <= 0)))[0]
        # now loop over the absorbers and add the tabulated voigt profiles
        for i, j, k in zip(ii[ll], jj[ll], ll):
            tau_lam[w1[k] : w2[k]] += c_voigt[k] * self.voigt_tab[i][j][x1[k] : x2[k]]
        return tau_lam


def fast_sum_of_voigts(
    wave, tau_lam, c_voigt, a, lambda_z, b, tauMin, tauMax, tauSplit
):
    """
    Given arrays of parameters, compute the summed optical depth
    spectrum of absorbers using Voigt profiles.
    Uses the Tepper-Garcia 2006 approximation for the Voigt function
    for large optical depth systems (defined by tauSplit), and
    a lookup table for low optical depth systems.
    """
    voigttab = VoigtTable.Instance(wave)
    # split out strong absorbers and do full calc
    ii = np.where(c_voigt >= tauSplit)[0]
    tau_lam = sum_of_voigts(
        wave, tau_lam, c_voigt[ii], a[ii], lambda_z[ii], b[ii], tauMin, tauMax
    )
    ii = np.where(c_voigt < tauSplit)[0]
    tau_lam = voigttab.sum_of_voigts(a[ii], b[ii], lambda_z[ii], c_voigt[ii], tau_lam)
    return tau_lam


def calc_tau_lambda(wave, los, **kwargs):
    """
    Compute the absorption spectrum, in units of optical depth, for
    a series of absorbers along a line-of-sight (los).
    """
    lymanseries_range = kwargs.get("lymanseries_range", default_lymanseries_range)
    tauMax = kwargs.get("tauMax", 15.0)
    tauMin = kwargs.get("tauMin", 1e-5)
    tau_lam = kwargs.get("tauIn", np.zeros_like(wave))
    fast = kwargs.get("fast", True)
    tauSplit = kwargs.get("fast_tauSplit", 1.0)
    # arrays of absorber properties
    NHI = 10 ** los["logNHI"]
    z1 = 1 + los["z"]
    b = los["b"]
    # first apply continuum blanketing. the dense systems will saturate
    # a lot of the spectrum, obviating the need for calculations of
    # discrete transition profiles
    tau_lam = sum_of_continuum_absorption(wave, tau_lam, NHI, z1, tauMin, tauMax)
    # now loop over Lyman series transitions and add up Voigt profiles
    for transition in range(*lymanseries_range):
        # transition properties
        lambda0, F, Gamma = transitionParams[transition]
        # Doppler width
        nu_D = b / (lambda0 * 1e-13)
        # Voigt a parameter
        a = Gamma / (fourpi * nu_D)
        # wavelength of transition at absorber redshift
        lambda_z = lambda0 * z1
        # coefficient of absorption strength (central tau)
        c_voigt = 0.014971475 * NHI * F / nu_D
        # all the values used to calculate tau, now just needs line profile
        if fast:
            tau_lam = fast_sum_of_voigts(
                wave, tau_lam, c_voigt, a, lambda_z, b, tauMin, tauMax, tauSplit
            )
        else:
            tau_lam = sum_of_voigts(
                wave, tau_lam, c_voigt, a, lambda_z, b, tauMin, tauMax
            )
    return tau_lam


class IGMTransmissionGrid(object):
    # TODO: Change documentation style
    # TODO: Fix documentation, does not reflect the current state of the function
    # TODO: (Future though) improve the IGM model, will require loads of time and literature scouting
    """
    Generate a library of forest transmission spectra, by mapping an array
    of emission redshifts to a set of sightlines.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Input wavelength grid (must be at fixed resolution!).
    z_em : `~numpy.ndarray`
        Array containing emission redshifts.
    nlos : int
        Number of lines-of-sight to generate.
    losMap : sequence
        Optional mapping from z_em to LOS. Must have the same number of
        elements and be in the range 0..nlos-1.
        If not provided and nlos>0, losMap is randomly generated.

    Returns
    -------
    spectra: dict
    T : `~numpy.ndarray`
        transmission spectra with shape (N(z),N(wave))
    z : `~numpy.ndarray`
        emission redshift for each spectrum
    losMap : `~numpy.ndarray`
        map of z_em <-> line-of-sight
    wave : `~numpy.ndarray`
        input wavelength grid
    voigtcache : bool
        use a lookup table of Voigt profiles to speed computation (def: True)
    """

    def __init__(self, wave, forest_model, num_sight_lines, **kwargs):
        self.spec_wave = wave
        self.forest_model = forest_model
        self.num_sight_lines = num_sight_lines
        self.verbose = kwargs.get("verbose", 0)
        self.no_sort_zz = kwargs.get("no_sort_z", False)
        self.subsample = kwargs.get("subsample", True)
        self.seed = kwargs.get("seed", 42)  # Default value for reproducibility
        self.voigtkwargs = {"fast": kwargs.pop("voigtcache", True)}

        # pad the lower redshift by just a bit
        self.z_min = wave.min() / 1215.7 - 1.01
        self.z_max = kwargs.get("z_max", 10)

        # Generate the lines-of-sight first, to preserve random generator order
        # TODO: use logging for these!
        if self.verbose:
            print("Generating {} sightlines".format(self.num_sight_lines))
            if self.verbose > 1:
                print("... using random seed {}".format(self.seed))

        # Changed: random seed is now fixed.
        # Also use generator instead of seed as per recommendation
        self.rng = np.random.default_rng(seed=self.seed)

        # This returns (z, logNHI, b) for each absorption system, randomly generated.
        print("[Info] Generating LOS.")
        self.sight_lines = [
            generate_los(self.forest_model, self.z_min, self.z_max, self.rng)
            for _ in tqdm(range(self.num_sight_lines))
        ]

        # default is 10 km/s
        forest_R_min = kwargs.get("R_min", 3e4)

        logwave = np.log(wave)
        dloglam = np.diff(logwave)

        if not np.allclose(dloglam, dloglam[0]):
            raise ValueError("Must have constant dloglam")

        spec_R = dloglam[0] ** -1
        self.n_rebin = int(np.ceil(forest_R_min / spec_R))
        self.forest_R = spec_R * self.n_rebin

        # go a half pixel below the minimum wavelength
        wave_min = np.exp(logwave[0] - 0.5 / spec_R)
        # go well beyond LyA to get maximum wavelength
        wave_max = min(wave[-1], 1250 * (1 + self.z_max))

        self.n_spec_pix = np.searchsorted(wave, wave_max, side="right")

        # now make sure it is an integer multiple
        wave_max = wave[self.n_spec_pix - 1]
        self.n_pix = self.n_spec_pix * self.n_rebin
        self.forest_wave = np.exp(
            np.log(wave_min) + self.forest_R**-1 * np.arange(self.n_pix)
        )
        dloglam = self.forest_R**-1

        # Make the wavelength grid
        # As far as I understand this is all to make a wavelength grid that is constant in resolution
        #  and is wide enough.
        self.forest_wave = np.exp(np.log(wave_min) + dloglam * np.arange(self.n_pix))

        self.tau = np.zeros(self.n_pix)
        if not self.subsample:
            self.all_T = []
        self.reset()

    def reset(self):
        self.current_sight_line_num = -1

    def next_spec(self, sight_line, z):
        if self.current_sight_line_num != sight_line:
            if self.verbose > 1:
                print("finished sightline ", self.current_sight_line_num)

            self.current_sight_line = self.sight_lines[sight_line]
            self.current_sight_line_num = sight_line
            self.tau[:] = 0.0
            self.zi = 0
        zi1 = self.zi
        los = self.current_sight_line
        zi2 = np.searchsorted(los["z"], min(z, self.z_max))
        if self.verbose > 1:
            print("extending sightline {} to z={:.4f}".format(sight_line, z))
        if zi2 < zi1:
            raise ValueError("must generate sightline in increasing redshift")
        self.zi = zi2
        tau = calc_tau_lambda(
            self.forest_wave, los[zi1:zi2], tau_in=self.tau, **self.voigtkwargs
        )
        T = np.exp(-tau).reshape(-1, self.n_rebin).mean(axis=1)
        self.T = T.astype(np.float32)
        if not self.subsample:
            self.all_T.append(self.T)
        return self.T

    def current_spec(self, sight_line):
        if self.subsample:
            return self.T
        else:
            return self.all_T[sight_line]

    def all_spec(self, los_map, z_em, **kwargs):
        if len(los_map) != len(z_em):
            raise ValueError
        if self.no_sort_z:
            z_i = np.arange(len(z_em))
        else:
            z_i = z_em.argsort()

        T = np.vstack([self.next_spec(los_map[i], z_em[i], **kwargs) for i in z_i])
        return Table(
            dict(
                T=T[z_i.argsort()].astype(np.float32),
                z=z_em.astype(np.float32),
                sight_line=los_map.astype(np.int32),
            )
        )

    def write(
        self, file_name, output_dir, t_spec=None, los_map=None, z_em=None, **kwargs
    ):
        """Save transmission spectra to a FITS file."""
        if t_spec is None:
            if los_map is None or z_em is None:
                raise ValueError("Must pass losMap and z")
            t_spec = self.all_spec(los_map, z_em, **kwargs)

        logwave = np.log(self.specWave[:2])
        dloglam = np.diff(logwave)

        # header keys
        t_spec.meta["CD1_1"] = float(dloglam)
        t_spec.meta["CRPIX1"] = 1
        t_spec.meta["CRVAL1"] = logwave[0]
        t_spec.meta["CRTYPE1"] = "LOGWAVE"
        t_spec.meta["IGMNLOS"] = self.num_sight_lines
        t_spec.meta["IGMMODL"] = str(self.forest_model)
        t_spec.meta["IGMRES"] = self.forest_R

        for k, v in kwargs.get("meta", {}).items():
            t_spec.meta[k] = v

        if not file_name.endswith(".fits") or file_name.endswith(".fits.gz"):
            file_name += ".fits"

        t_spec.write(
            os.path.join(output_dir, file_name),
            overwrite=kwargs.get("overwrite", False),
        )


# for now just duck-typing this
class CachedIGMTransmissionGrid(object):
    def __init__(self, fileName, outputDir="."):
        if not (fileName.endswith(".fits") or fileName.endswith(".fits.gz")):
            fileName += ".fits"
        fn = os.path.join(outputDir, fileName)
        self.tspec = tspec = Table.read(fn)
        hdr = fits.getheader(fn, 1)
        nwave = tspec["T"].shape[1]
        wi = np.arange(nwave)
        logwave = hdr["CRVAL1"] + hdr["CD1_1"] * (wi - (hdr["CRPIX1"] - 1))
        self.specWave = exp(logwave)
        self.numSightLines = hdr["IGMNLOS"]
        self.losIndex = {
            tuple(losNum_z): i for i, losNum_z in enumerate(tspec["sightLine", "z"])
        }
        self.losMap = self.tspec["sightLine"]

    def next_spec(self, sightLine, z, **kwargs):
        return self.current_spec(sightLine, z, **kwargs)

    def current_spec(self, sightLine, z, **kwargs):
        # z is saved as float32 and need to match type
        i = self.losIndex[(sightLine, np.float32(z))]
        return self.tspec["T"][i]


def generate_binned_forest(
    fileName, forestModel, nlos, zbins, waverange, R, outputDir=".", **kwargs
):
    wave = fixed_R_dispersion(*tuple(waverange + (R,)))
    z = np.tile(zbins[:, np.newaxis], nlos).transpose()
    ii = np.arange(nlos)
    losMap = np.tile(ii[:, np.newaxis], len(zbins))
    fGrid = IGMTransmissionGrid(wave, forestModel, nlos, **kwargs)
    tspec = fGrid.all_spec(losMap.ravel(), z.ravel())
    if fileName is None:
        return tspec
    else:
        fGrid.write(
            fileName,
            outputDir,
            tspec=tspec,
            meta={"ZBINS": ",".join(["%.3f" % _z for _z in zbins])},
        )


def _get_forest_mags(forestModel, zbins, waverange, R, photoMap, n, **kwargs):
    wave = fixed_R_dispersion(*tuple(waverange + (R,)))
    grid = generate_binned_forest(None, forestModel, n, zbins, waverange, R, **kwargs)
    nBands = len(photoMap.getBandpasses())
    #
    fGrid = grid.group_by("sightLine")
    wi = np.arange(fGrid["T"].shape[-1], dtype=np.float32)
    fGrid["dmag"] = np.zeros((1, nBands), dtype=np.float32)
    fGrid["fratio"] = np.zeros((1, nBands), dtype=np.float32)
    #
    fakespec = namedtuple("fakespec", "wave,f_lambda")
    refspec = fakespec(wave, np.ones_like(wave))
    refmags, reffluxes = photoMap.calcSynPhot(refspec)
    #
    for snum, sightLine in zip(fGrid.groups.keys["sightLine"], fGrid.groups):
        for i, z in enumerate(zbins):
            spec = fakespec(wave, sightLine["T"][i])
            mags, fluxes = photoMap.calcSynPhot(spec)
            dmag = mags - refmags
            dmag[fluxes <= 0] = 99
            sightLine["dmag"][i] = dmag
            sightLine["fratio"][i] = fluxes.clip(0, np.inf) / reffluxes
        if ((snum + 1) % 10) == 0:
            try:
                pid = multiprocessing.current_process().name.split("-")[1]
            except:
                pid = "--"
            print("[%2s] completed %d sightlines" % (pid, snum + 1))
    del fGrid["z", "T"]
    return fGrid


def generate_grid_forest(
    fileName,
    forestModel,
    nlos,
    zbins,
    waverange,
    R,
    photoMap,
    outputDir=".",
    nproc=1,
    **kwargs,
):
    n = nlos // nproc
    if nproc == 1:
        _map = map
    else:
        pool = multiprocessing.Pool(nproc)
        _map = pool.map
    forest_generator = partial(
        _get_forest_mags, forestModel, zbins, waverange, R, photoMap, **kwargs
    )
    _nlos = np.repeat(n, nproc)
    _nlos[-1] += nlos - np.sum(_nlos)
    fGrids = _map(forest_generator, _nlos)
    for i in range(1, len(fGrids)):
        fGrids[i]["sightLine"] += fGrids[i - 1]["sightLine"].max() + 1
    fGrid = vstack(fGrids)
    fGrid.meta["ZBINS"] = ",".join(["%.3f" % _z for _z in zbins])
    fGrid.meta["BANDS"] = ",".join(photoMap.getBandpasses())
    fGrid.write(os.path.join(outputDir, fileName), overwrite=True)
    if nproc > 1:
        pool.close()


class GridForest(object):
    def __init__(self, fileName, simBands, median=False):
        self.simBands = np.array(simBands)
        self.data = Table.read(fileName).group_by("sightLine")
        self.numSightLines = len(self.data.groups)
        zbins = self.data.meta["ZBINS"].split(",")
        self.zbins = np.array(zbins).astype(np.float32)
        self.dz = np.diff(self.zbins)
        self.bands = np.array(self.data.meta["BANDS"].split(","))
        self.ii = np.array([np.where(b == self.simBands)[0][0] for b in self.bands])
        shp = (self.numSightLines, len(self.zbins), -1)
        self.dmag = np.array(self.data["dmag"]).reshape(shp)
        self.frat = np.array(self.data["fratio"]).reshape(shp)
        if median:
            self.dmag = np.median(self.dmag, axis=0)[None, :, :]
            self.frat = np.median(self.frat, axis=0)[None, :, :]
            self.numSightLines = 1
        self.dmdz = np.diff(self.dmag, axis=1) / self.dz[:, None]
        self.dfdz = np.diff(self.frat, axis=1) / self.dz[:, None]

    def get(self, losNum, z):
        zi = np.digitize(z, self.zbins) - 1
        if np.any((zi < 0) | (zi >= len(self.zbins) - 1)):
            print(
                "WARNING: qso z range {:.3f}|{:.3f} ".format(z.min(), z.max()), end=" "
            )
            print(
                "outside forest grid {:.3f}|{:.3f}".format(
                    self.zbins[0], self.zbins[-1]
                )
            )
            zi = zi.clip(0, len(self.zbins) - 2)
        dz = z - self.zbins[zi]
        dmag = self.dmag[losNum, zi] + self.dmdz[losNum, zi] * dz[:, None]
        frat = self.frat[losNum, zi] + self.dfdz[losNum, zi] * dz[:, None]
        return self.ii, dmag, frat


# for now just duck-typing this
class MeanIGMTransmissionGrid(object):
    def __init__(self, fileName, wave, outputDir="."):
        if not fileName.endswith(".fits") or fileName.endswith(".fits.gz"):
            fileName += ".fits"
        self.outWave = wave
        fn = os.path.join(outputDir, fileName)
        tspec = Table.read(fn)
        hdr = fits.getheader(fn, 1)
        nwave = tspec["T"].shape[1]
        wi = np.arange(nwave)
        logwave = hdr["CRVAL1"] + hdr["CD1_1"] * (wi - (hdr["CRPIX1"] - 1))
        self.specWave = exp(logwave)
        self.wi = np.searchsorted(self.outWave, self.specWave[-1])
        nlos = hdr["IGMNLOS"]
        self.numSightLines = 1
        self.zBins = np.array(list(map(float, hdr["ZBINS"].split(","))))
        self.meanT = tspec["T"].reshape(nlos, -1, nwave).mean(axis=0)

    def next_spec(self, sightLine, z, **kwargs):
        return self.spec(z)

    def current_spec(self, sightLine, z, **kwargs):
        return self.spec(z)

    def spec(self, z):
        zi = np.searchsorted(self.zBins, z)
        T = self.meanT[zi].clip(0, 1)  # XXX clip is there for bad vals
        T = resample(self.specWave, T, self.outWave[: self.wi])
        return T
