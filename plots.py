from chainconsumer import ChainConsumer
import numpy as np
import matplotlib.pyplot as plt
import utils
import os

def get_max(ensemble,param):
    """
    Returns the max-posterior sample.
    """
    amax = np.argmax(ensemble.get_log_posterior().flatten())
    samp = ensemble.get_chain()[param]
    evals, temps, walker, nleaves,dim = samp.shape
    samp = samp.reshape(evals*temps*walker,nleaves,dim)[amax]
    naninds = np.logical_not(np.isnan(samp))
    nleaves = np.sum(naninds[:,0])
    return samp[naninds].reshape(nleaves,dim)

def get_clean_best_nleave_chain(ensemble, param, temp=0):
    """
    Returns only the chains with the same number of leaves for "param" as the max-posterior one.
    """
    amax = np.argmax(ensemble.get_log_posterior())
    best_nleaves = ensemble.get_nleaves()[param].flatten()[amax]
    mask = ensemble.get_nleaves()[param][:,temp,:] == best_nleaves
    sparams = ensemble.get_chain()[param][:,temp,:][mask]

    naninds = np.logical_not(np.isnan(sparams))
    num,_,dim = sparams.shape
    return sparams[naninds].reshape(num,best_nleaves,dim)

def get_clean_chain(ensemble, param, temp=0):
    """Simple utility function to extract the squeezed chains for all the parameters
    """
    coords = ensemble.get_chain()[param]
    _, ntemps, nwalkers, _, ndim = coords.shape
    naninds    = np.logical_not(np.isnan(coords[:, temp, :, :, 0].flatten()))
    samples_in = np.zeros((coords[:, temp, :, :, 0].flatten()[naninds].shape[0], ndim))  # init the chains to plot
    # get the samples to plot
    for d in range(ndim):
        givenparam = coords[:, temp, :, :, d].flatten()
        samples_in[:, d] = givenparam[
            np.logical_not(np.isnan(givenparam))
        ]  # Discard the NaNs, each time they change the shape of the samples_in
    return samples_in

def color_interp(rgb_0, rgb_f, steps):
    """
    Gives good colors for the dif temps on the chains.
    """
    r, g, b = rgb_0
    re, ge, be = rgb_f
    return [
        "#{:02X}{:02X}{:02X}".format(
            int((re - r) / steps * i + r),
            int((ge - g) / steps * i + g),
            int((be - b) / steps * i + b),
        )
        for i in range(steps)
    ]

def corner_plot(samples,param_names, outdir="./plots"):
    c = ChainConsumer()
    c.add_chain(samples, parameters=param_names, color='orange')
    c.configure(bar_shade=True, plot_hists=False)
    fig = c.plotter.plot(figsize=(4, 4))
    plt.savefig(os.path.join(outdir, f"corner.png"))
    plt.close()

def trace_plots(ensemble,outdir="./plots",outfile = "test.png"):
    outfile = outfile.split(".")
    # trace plots for one of the branches
    ll = ensemble.get_log_posterior()
    branches = ensemble.branch_names
    for branch_name in branches:
        samples_burnt = ensemble.get_chain()[branch_name]
        _, ntemps, nwalkers, _, ndim = samples_burnt.shape
        fig, axes = plt.subplots(ndim + 1, figsize=(10, 7 / 3 * (ndim + 1)), sharex=True)
        csteps = color_interp((0, 10, 200), (200, 10, 0), ntemps)
        for j in range(ndim):
            ax = axes[j]
            ax.set_xlim(0, samples_burnt.shape[0])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            for t in reversed(range(ntemps)):
                for k in range(nwalkers):
                    ax.scatter(range(samples_burnt.shape[0]), samples_burnt[:, t, k, 0, j], c=csteps[t], alpha=0.2,s=0.5)

        for k in range(nwalkers):
            axes[-1].plot(ll[:, 0, k])
        axes[-1].set_ylabel("log pos (T=1)")
        axes[-1].yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.savefig(os.path.join(outdir, f"{outfile[0]}_{branch_name}_trace.{outfile[1]}"))
        plt.close()

def plot_best_nleaves(wl,flux, ensemble,outdir="./plots"):
    plt.plot(wl, flux, label="data", color="lightskyblue")
    knots = get_clean_best_nleave_chain(ensemble, 'knots', temp=0)
    edges = get_clean_best_nleave_chain(ensemble, 'edges', temp=0)
    gauss = get_clean_best_nleave_chain(ensemble, 'gauss', temp=0)
    for i in range(100):
        gaussians = utils._multi_gaussian(wl, gauss[i])
        interp_mod = utils.get_spline(knots[i], edges[i], wl)
        plt.plot(wl, interp_mod(wl) + gaussians, alpha=0.02, color="orange")
    gaussians = utils._multi_gaussian(wl, get_max(ensemble,'gauss'))
    interp_mod = utils.get_spline(get_max(ensemble, 'knots'), get_max(ensemble, 'edges'), wl)
    plt.plot(wl, interp_mod(wl), color="grey",label='spline')
    plt.plot(wl, interp_mod(wl) + gaussians, color="black",label='spline+gaussians')
    plt.savefig(os.path.join(outdir, f"best_samples.png"))
    plt.close()

def get_leaves(ensemble,branchname,temp=0):
    branch = ensemble.get_chain()[branchname]
    nsteps, ntemps, nwalkers, nmaxleaves, ndim = branch.shape
    branch = branch[:,temp,:,:,:].reshape(nsteps*nwalkers,nmaxleaves,ndim)
    mask = ensemble.get_inds()[branchname][:,temp,:,:]
    mask = mask.reshape(nsteps*nwalkers,nmaxleaves)
    leaves = []
    for step in np.arange(len(mask)):
        leaves.append(branch[step][mask[step]])
    return leaves

def plot_best(wl,flux, ensemble, outdir="./plots", outfile = "test.png"):
    outfile = outfile.split(".")
    plt.plot(wl, flux, label="data", color="lightskyblue")
    knots = get_leaves(ensemble,"knots",temp=0)
    edges = get_leaves(ensemble, 'edges', temp=0)
    gauss = get_leaves(ensemble, 'gauss', temp=0)
    logl = ensemble.get_log_posterior()[:,0,:].flatten()
    for i in range(len(logl)):
        gaussians = utils._multi_gaussian(wl, gauss[i])
        interp_mod = utils.get_spline(knots[i], edges[i], wl)
        plt.plot(wl, interp_mod(wl) + gaussians, alpha=0.02, color="orange")
        plt.plot(wl, interp_mod(wl), alpha=0.02, color="green")
    #gaussians = utils._multi_gaussian(wl, get_max(ensemble,'gauss'))
    #interp_mod = utils.get_spline(get_max(ensemble, 'knots'), get_max(ensemble, 'edges'), wl)
    #plt.plot(wl, interp_mod(wl), color="grey",label='spline')
    #plt.plot(wl, interp_mod(wl) + gaussians, color="black",label='spline+gaussians')
    #plt.plot(wl, np.median(interp_mod(wl)) + gaussians, color="red",label='gaussians')
    plt.savefig(os.path.join(outdir, f"{outfile[0]}_best_samples.{outfile[1]}"))
    plt.close()
    return interp_mod
