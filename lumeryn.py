import numpy as np
import configparser
from scipy.signal import find_peaks,peak_prominences,peak_widths,savgol_filter
from eryn.prior import uniform_dist
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.moves import GaussianMove,GroupMove
from eryn.backends import HDFBackend
from spec_likelihood import SpectraLikelihood
import utils
import plots
import matplotlib.pyplot as plt
import os
import user_prior

class LumerynSpecFitter:
    def __init__(self, wl, flux, eflux, prior_conf=None, sampler_conf=None):
        self.lumdir = os.path.dirname(os.path.realpath(__file__))
        if prior_conf is None:
            prior_conf = self.lumdir+'/prior.ini'
            print(f"[lumeryn]: Loading default priors from {prior_conf}")
        if sampler_conf is None:
            sampler_conf = self.lumdir+'/sampler.ini'
            print(f"[lumeryn]: Loading default sampler params from {sampler_conf}")
        prior_par = configparser.ConfigParser()
        mcmc_par = configparser.ConfigParser()
        prior_par.read(prior_conf)
        mcmc_par.read(sampler_conf)
        self.prior_par = prior_par
        self.priors = user_prior.init_prior(wl,flux,eflux)
        self.mcmc_par = mcmc_par
        mcmc_par = self.mcmc_par['eryn']
        self.ntemps = int(mcmc_par['ntemps'])
        self.nwalkers = int(mcmc_par['nwalkers'])
        self.wl = wl
        self.maxflux = max(flux)
        self.flux = flux#/self.maxflux
        self.eflux= eflux#/self.maxflux

    def generate_initial_guess(self, nameplot=None, sfwindow=50, prominence=None, knots_continuum=6):
        """
        Initializes the MCMC for spectral fitting of absorption lines and continuum.
        For the gaussians:
        - Smooths the data using a Savitzky-Golay filter. Specify window size and pixel width min and max.
        - Identifies peaks and their parameters.
        For the continuum:
        - Smooths the data to a spline with "n" defined knots to give the continuum spline a start.
        """
        wl = self.wl
        dwl=wl[1]-wl[0]
        flux = self.flux
        eflux = self.eflux
        priors = self.priors
        s, ss = utils._binary_search(knots_continuum, wl, flux,eflux)
        smooth_flux = savgol_filter(flux, sfwindow, 3)
        priors2 = user_prior.init_prior(np.arange(len(smooth_flux)),-smooth_flux, eflux)
        #import pdb
        #pdb.set_trace()
        if prominence is None:
            prominence = [-priors2['gauss'][0].max_val, -priors2['gauss'][0].min_val] # Using half of signal/noise as prominence enough to id.
        pixwidth = [priors2['gauss'][2].min_val,priors2['gauss'][2].max_val]
        #apeak, properties = find_peaks(-smooth_flux/ss(wl)+1, width=pixwidth, prominence=prominence)
        apeak, properties = find_peaks(-smooth_flux, width=pixwidth, prominence=prominence)
        #prominences = properties['prominences']*ss(wl[apeak])
        prominences = properties['prominences']#*ss(wl[apeak])
        x1 = properties['left_ips']
        x2 = properties['right_ips']

        widths = wl[np.rint(x2).astype(int)] - wl[np.rint(x1).astype(int)]
        #prominences are negative for absorption lines
        if nameplot is not None:
            plt.plot(wl, flux, label="data", color="lightskyblue")
            plt.plot(wl, smooth_flux)
            plt.plot(wl[apeak], smooth_flux[apeak], "x")
            plt.vlines(x=wl[apeak], ymin=smooth_flux[apeak], ymax= smooth_flux[apeak]+prominences, color='black')
            plt.hlines(smooth_flux[apeak]+prominences/2, wl[np.rint(x1).astype(int)], wl[np.rint(x2).astype(int)], color="C2")
            plt.plot(wl, ss(wl), label='continuum start', color='yellow')
            plt.ylabel("flux")
            plt.xlabel("wavelength")
            plt.legend()
            plt.show()
            plt.savefig(nameplot)
        self._xpeak = wl[apeak]
        self._ypeak = prominences
        self._wpeak = widths
        self._knots = ss

        return


    def initialize_chains(self):
        wl = self.wl
        flux = self.flux
        eflux = self.eflux
        if not hasattr(self,'_xpeak'):
            self.generate_initial_guess()
        xpeak = self._xpeak
        ypeak = self._ypeak
        wpeak = self._wpeak
        knots = self._knots
        prior_par = self.prior_par
        mcmc_par = self.mcmc_par['eryn']
        ntemps = self.ntemps
        nwalkers = self.nwalkers
        priors = self.priors

        branches = prior_par.sections()
        ndims = {}
        coords = {}
        indxs = {}
        groups = {}
        nleaves_max = {}
        nleaves_min = {}


        for branch in branches:
            ndims[branch]  = int(prior_par[branch]['dim'])
            nleaves_max[branch]  = int(prior_par[branch]['nmax'])
            nleaves_min[branch]  = int(prior_par[branch]['nmin'])
            coords[branch] = np.zeros((ntemps, nwalkers, int(prior_par[branch]['nmax']), int(prior_par[branch]['dim'])))
            indxs[branch]  = np.zeros((ntemps, nwalkers, int(prior_par[branch]['nmax'])), dtype=bool)
            groups[branch] = np.arange(coords[branch].shape[0] * coords[branch].shape[1]).reshape(coords[branch].shape[:2])[:, :, None]
            groups[branch] = np.repeat(groups[branch], coords[branch].shape[2], axis=-1)

        def init_check(coords):
            """Checking the initial values are within the priors
            """
            for name in prior_par.sections():
                ndim = int(prior_par[name]['dim'])
                nleaf = int(prior_par[name]['nmax'])
                prior_bounds = np.array([ [priors[name][n].min_val, priors[name][n].max_val] for n in range(ndim)])
                mn,mx = prior_bounds.T
                mask = np.logical_and.reduce((coords[name] >= mn) & (coords[name] <= mx),axis=-1)
                if not mask.all():
                    print(f"[lumeryn]: error, some {name} initial conditions outside priors")
                    print(f"[lumeryn]: priors: {prior_bounds}")
                    print("[lumeryn]: bad coords:")
                    print(coords[name][~mask])
            return

        def clamp(x,minval,maxval):
            x=np.atleast_2d(x)
            x[x<minval] = minval
            x[x>maxval] = maxval
            return x

        for w in range(nwalkers):
            for t in range(ntemps):
                # gauss
                for nn in range(int(prior_par["gauss"]["nmax"])):
                    i = np.random.randint(0, len(ypeak))
                    ball_radius = [(priors["gauss"][n].max_val - priors["gauss"][n].min_val)/10000 for n in range(3)]
                    coords["gauss"][t, w, nn, 0] = -ypeak[i] + ball_radius[0] * np.random.uniform(-1,1)
                    coords["gauss"][t, w, nn, 1] = xpeak[i] + ball_radius[1] * np.random.uniform(-1,1)
                    coords["gauss"][t, w, nn, 2] = wpeak[i] + ball_radius[2] * np.random.uniform(-1,1)
                    coords["gauss"][t, w, nn, 0]=clamp(coords["gauss"][t, w, nn, 0], priors['gauss'][0].min_val,priors['gauss'][0].max_val)
                    coords["gauss"][t, w, nn, 1]=clamp(coords["gauss"][t, w, nn, 1], priors['gauss'][1].min_val,priors['gauss'][1].max_val)
                    coords["gauss"][t, w, nn, 2]=clamp(coords["gauss"][t, w, nn, 2], priors['gauss'][2].min_val,priors['gauss'][2].max_val)
                for nn in range(int(prior_par["knots"]["nmax"])):
                    i = np.random.randint(1, len(knots.get_knots()) - 1)
                    ball_radius = [(priors["knots"][n].max_val - priors["knots"][n].min_val)/10000 for n in range(2)]
                    coords["knots"][t, w, nn, 0] = knots.get_knots()[i] + ball_radius[0] * np.random.uniform(-1,1)
                    coords["knots"][t, w, nn, 1] = knots(knots.get_knots())[i] + ball_radius[1] * np.random.uniform(-1,1)
                    clamp(coords["knots"][t, w, nn, 0], priors['knots'][0].min_val,priors['knots'][0].max_val)
                    clamp(coords["knots"][t, w, nn, 1], priors['knots'][1].min_val,priors['knots'][1].max_val)

                for nn in range(int(prior_par["edges"]["nmax"])):
                    ball_radius = [(priors["edges"][n].max_val - priors["edges"][n].min_val)/10000 for n in range(2)]
                    coords["edges"][t, w, nn, 0] = knots(knots.get_knots())[0] + ball_radius[0] * np.random.uniform(-1,1)
                    coords["edges"][t, w, nn, 1] = knots(knots.get_knots())[-1] + ball_radius[1] * np.random.uniform(-1,1)
                    clamp(coords["edges"][t, w, nn, 0], priors['edges'][0].min_val,priors['edges'][0].max_val)
                    clamp(coords["edges"][t, w, nn, 1], priors['edges'][1].min_val,priors['edges'][1].max_val)

        init_check(coords)

        # turn on at least this many knots in each temp and walker
        indxs['knots'][:, :, :int(prior_par['knots']['min_on'])] = True
        # turn on at least this many gaussians in each temp and walker
        indxs['gauss'][:, :, :int(prior_par['gauss']['min_on'])] = True

        for name, inds_temp in indxs.items():
            inds_fix = np.where(np.sum(inds_temp, axis=-1) == 0)

            for ind1, ind2 in zip(inds_fix[0], inds_fix[1]):
                inds_temp[ind1, ind2, 0] = True

        coords_in = {name: coords[name][indxs[name]] for name in coords}
        groups_in = {name: groups[name][indxs[name]] for name in groups}
        self.coords_in = coords_in
        self.groups_in = groups_in
        self.indxs = indxs
        self.coords = coords
        self.ndims = ndims
        self.nleaves_max = nleaves_max
        self.nleaves_min = nleaves_min
        return

    def fit(self,backendname=None):
        if backendname is None:
            print("[lumeryn]: No backend file provided, not saving chains.")
        wl = self.wl
        flux = self.flux
        eflux = self.eflux
        if not hasattr(self,'coords_in'):
            self.initialize_chains()
        prior_par = self.prior_par
        mcmc_par = self.mcmc_par['eryn']
        ntemps = self.ntemps
        nwalkers = self.nwalkers
        priors = self.priors
        coords_in = self.coords_in
        groups_in = self.groups_in
        indxs = self.indxs
        coords = self.coords

        branches = prior_par.sections()

        speclike = SpectraLikelihood(wl, flux, eflux, wltol=.05, kind='linear')

        ll_eval = speclike.evaluate([coords_in['gauss'], coords_in["knots"], coords_in["edges"]], [groups_in['gauss'], groups_in["knots"], groups_in["edges"]])

        #print('llh shape = ', ll_eval.shape)
        #print('ll_eval = ', ll_eval)

        log_prob = ll_eval.reshape(ntemps, nwalkers)

        betas  = np.linspace(1.0, 0.0, ntemps)
        factor = float(mcmc_par['factor'])
        cov    = {
            "gauss": np.diag(np.ones(3)) * factor,
            "knots": np.diag(np.ones(2)) * factor,
            "edges": np.diag(np.ones(2)) * factor
        }

        moves = GaussianMove(cov)

        if backendname is not None:
            backend = HDFBackend(backendname)
        else:
            backend = None

        ensemble = EnsembleSampler(
            nwalkers,
            self.ndims,  # assumes ndim_max
            speclike.evaluate,
            priors,
            args=None,
            tempering_kwargs=dict(betas=betas,adaptive=False),
            nbranches=len(self.nleaves_max),
            branch_names=list(self.nleaves_max.keys()),
            nleaves_max=self.nleaves_max,
            nleaves_min=self.nleaves_min,
            provide_groups=True,
            update_iterations=1,
            plot_iterations=-1,
            moves=moves,
            rj_moves=True,
            vectorize=True,
            backend=backend,
        )

        print('[lumeryn]:  * Started sampling ...')
        state = State(coords, log_like=log_prob, betas=betas, blobs=None, inds=indxs)

        burnin = int(mcmc_par['burnin'])
        nsteps = int(mcmc_par['nsteps'])
        ensemble.run_mcmc(state, nsteps, burn=burnin, progress=True, thin_by=1)

        print('[lumeryn]:  * Finished sampling!')
        return ensemble

if __name__ == "__main__":
    datafile = "injectiontest.dat"
    fname = datafile.split(".")[0]
    wl,flux,eflux = np.loadtxt(datafile).T
    lum = LumerynSpecFitter(wl,flux,eflux)

    # These two following routines will be run by default when calling LumerynSpecFitter.fit(), but are accesible and have many useful options.
    lum.generate_initial_guess(nameplot='initialization.pdf', sfwindow=20)#, pixwidthmin=5, pixwidthmax=len(wl)/5)
    lum.initialize_chains()

    ensemble = lum.fit()#backendname = fname+".hd5")
    plots.plot_best(wl,flux,ensemble, outfile = fname+".png")
    plots.trace_plots(ensemble,outfile = fname+".png")


