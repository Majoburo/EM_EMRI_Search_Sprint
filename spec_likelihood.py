from scipy import interpolate
import numpy as np
import utils

class SpectraLikelihood2:

    def __init__(self, wavelength, flux, eflux, inf=1e14, expmax = 500, kind='linear', wltol=.1):
        """
        Gaussian likelihood model using a spline continuum and gaussians for absorption features.

        Parameters
        ----------
         wl : (ndarray) wavelength array of the spectrum, size nwl
       flux : (ndarray) flux array, size nwl
      eflux : (float) flux error array in standard deviations, size nwl
        inf : (float or np.inf) if the likelihood diverges, it will be set equal to - inf
     expmax : (float) maximum value allowed in the exponential function. If this is reached, the log-likelihood will return -infinity.
       kind : (string) Interpolation kind. Set to 'cubic' by default.
       wltol : (float [ 0 < wltol]) Tolerance on proximity of knots in wl
        """

        # Wavelenghts
        self.wl   = wavelength
        #self.maxflux = max(flux)
        self.flux = flux#/self.maxflux
        self.eflux = eflux#/self.maxflux
        # Interpolation kind
        self.kind = kind
        # tolerance on proximity of knots in logspace
        self.wltol = wltol
        self.inf = inf

    def get_spline(self, x, groups):
        """Get the spline model for the continuum, given some knots

        Parameters
        ----------
        x, groups : (ndarray) continuum parameters

        Returns
        -------
        spline : interpolate.interp1d evaluated
        """

        # I will consider two models. One handling the internal knots, and one for the edges
        internal_knots_parameters, control_points_edges = x
        group1, group2 = groups
        knots = internal_knots_parameters[:, 0]
        control_points = internal_knots_parameters[:, 1]

        num_groups = int(group1.max() + 1) #ntemps*nwalkers #int(group1.max() + 1)
        spline_model = np.empty((num_groups, len(self.wl)))
        spline_model[:] = np.nan

        # Loop over the temperatures vs walkers
        for i in range(num_groups):
            inds1 = np.where(group1 == i)
            if len(inds1[0]) == 0:
                continue
            knots_i = knots[inds1]
            control_points_i = control_points[inds1]

            inds2 = np.where(group2 == i)
            control_points_edges_i = np.squeeze(control_points_edges[inds2])

            # Remove zeros 
            knots_i  = knots_i[knots_i != 0.]
            control_points_i = control_points_i[control_points_i != 0.]

            knots_list  = np.array([self.wl[0]] + list(knots_i) + [self.wl[-1]])
            control_pts = np.array([control_points_edges_i[0]] + list(control_points_i) + [control_points_edges_i[-1]])

            # Control for knots very close to each other
            if not np.any(np.diff(np.array(knots_list)) < self.wltol):
                interp_model = interpolate.Akima1DInterpolator(knots_list, control_pts, axis=-1, method='makima')
                spline_model[i] = interp_model(self.wl)

                # To prevent overflow
                #if np.any(spline_model[i] > self.expmax):
                #    print('[Overflow!]')
                #    i_over = np.where((spline_model[i] > self.expmax) | (np.isnan(spline_model[i])))
                #    spline_model[i][i_over] = np.nan
        # Return the correct quantity
        return spline_model

    def combine_gaussians(self, wl, gparams, groups, num_groups):
        template = np.zeros((num_groups, wl.shape[0]))
        for i in range(num_groups):
            which_gauss = np.where(groups==i)[0]
            template[i,:] = sum(utils._gaussian(wl, *gparams[k,:]) for k in which_gauss)
        return template

    def evaluate(self, x, groups):
        """
        Calculate the log-likelihood.

        Parameters
        ----------
             params : (ndarray) vector of gauss and spline parameters.

        Returns
        -------
             ll : (ndarray) log-likelihood value.
        """
        #import pdb
        #pdb.set_trace()

        num_groups = int(groups[1].max() + 1)

        template = np.zeros((num_groups,self.wl.shape[0]))

        if x is not None and groups is not None:
            template += self.combine_gaussians(self.wl, x[0], groups[0],num_groups).reshape(template.shape)
            template += self.get_spline(x[1:], groups[1:])
        # Sum with the normalization factor here
        ## RECHECK THE NORMALIZATION IS CORRECT HERE!!!
        ll = np.nan_to_num(-.5 *(((template - self.flux) / self.eflux) ** 2 + np.log(2.*np.pi)) -np.log(self.eflux),nan=-np.inf)
        ll = np.sum(ll,axis=-1)
        ll[~np.isfinite(ll)] = -self.inf
        return ll

class SpectraLikelihood:

    def __init__(self, wavelength, flux, eflux, inf=1e14, expmax = 500, kind='linear', wltol=.1):
        """
        Gaussian likelihood model using a spline continuum and gaussians for absorption features.

        Parameters
        ----------
         wl : (ndarray) wavelength array of the spectrum, size nwl
       flux : (ndarray) flux array, size nwl
      eflux : (float) flux error array in standard deviations, size nwl
        inf : (float or np.inf) if the likelihood diverges, it will be set equal to - inf
     expmax : (float) maximum value allowed in the exponential function. If this is reached, the log-likelihood will return -infinity.
       kind : (string) Interpolation kind. Set to 'cubic' by default.
       wltol : (float [ 0 < wltol]) Tolerance on proximity of knots in wl
        """

        # Wavelenghts
        self.wl   = wavelength
        #self.maxflux = max(flux)
        self.flux = flux#/self.maxflux
        self.eflux = eflux#/self.maxflux
        # Interpolation kind
        self.kind = kind
        # tolerance on proximity of knots in logspace
        self.wltol = wltol
        self.inf = inf

    def get_spline(self, x, groups):
        """Get the spline model for the continuum, given some knots

        Parameters
        ----------
        x, groups : (ndarray) continuum parameters

        Returns
        -------
        spline : interpolate.interp1d evaluated
        """

        # I will consider two models. One handling the internal knots, and one for the edges
        internal_knots_parameters, control_points_edges = x
        group1, group2 = groups
        knots = internal_knots_parameters[:, 0]
        control_points = internal_knots_parameters[:, 1]

        num_groups = int(group1.max() + 1) #ntemps*nwalkers #int(group1.max() + 1)
        spline_model = np.empty((num_groups, len(self.wl)))
        spline_model[:] = np.nan

        # Loop over the temperatures vs walkers
        for i in range(num_groups):
            inds1 = np.where(group1 == i)
            if len(inds1[0]) == 0:
                continue
            knots_i = knots[inds1]
            control_points_i = control_points[inds1]

            inds2 = np.where(group2 == i)
            control_points_edges_i = np.squeeze(control_points_edges[inds2])

            # Remove zeros 
            knots_i  = knots_i[knots_i != 0.]
            control_points_i = control_points_i[control_points_i != 0.]

            knots_list  = np.array([self.wl[0]] + list(knots_i) + [self.wl[-1]])
            control_pts = np.array([control_points_edges_i[0]] + list(control_points_i) + [control_points_edges_i[-1]])

            # Control for knots very close to each other
            if not np.any(np.diff(np.array(knots_list)) < self.wltol):
                interp_model = interpolate.make_interp_spline(knots_list, control_pts, k=3, axis=-1)
                spline_model[i] = interp_model(self.wl)

                # To prevent overflow
                #if np.any(spline_model[i] > self.expmax):
                #    print('[Overflow!]')
                #    i_over = np.where((spline_model[i] > self.expmax) | (np.isnan(spline_model[i])))
                #    spline_model[i][i_over] = np.nan
        # Return the correct quantity
        return spline_model

    def combine_gaussians(self, wl, gparams, groups, num_groups):
        template = np.zeros((num_groups, wl.shape[0]))
        for i in range(num_groups):
            which_gauss = np.where(groups==i)[0]
            template[i,:] = sum(utils._gaussian(wl, *gparams[k,:]) for k in which_gauss)
        return template

    def evaluate(self, x, groups):
        """
        Calculate the log-likelihood.

        Parameters
        ----------
             params : (ndarray) vector of gauss and spline parameters.

        Returns
        -------
             ll : (ndarray) log-likelihood value.
        """
        #import pdb
        #pdb.set_trace()

        num_groups = int(groups[1].max() + 1)

        template = np.zeros((num_groups,self.wl.shape[0]))

        if x is not None and groups is not None:
            template += self.combine_gaussians(self.wl, x[0], groups[0],num_groups).reshape(template.shape)
            template += self.get_spline(x[1:], groups[1:])
        # Sum with the normalization factor here
        ## RECHECK THE NORMALIZATION IS CORRECT HERE!!!
        ll = np.nan_to_num(-.5 *(((template - self.flux) / self.eflux) ** 2 + np.log(2.*np.pi)) -np.log(self.eflux),nan=-np.inf)
        ll = np.sum(ll,axis=-1)
        ll[~np.isfinite(ll)] = -self.inf
        return ll

