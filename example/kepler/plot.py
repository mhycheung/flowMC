import matplotlib.pyplot as plt
import numpy as np
import corner
import jax
import jax.numpy as jnp

def draw_corner(chains, true_val, labels, n_last_spls=1000, labelpad=10.0,
                frac_bounds=0.95, n_plot_chains=10):
    plt.rcParams.update({'font.size': 8})
    n_dim = chains.shape[-1]
    figure = corner.corner(chains[:, -n_last_spls:, :].reshape(-1,n_dim),
                    labels=labels,
                    # var_names=var_names,
                    truths=true_val,
                    range=[frac_bounds] * n_dim,
                    labelpad=labelpad,
                    truth_color='g')
    figure.set_size_inches(7, 7)
    figure.suptitle('Visualize chains')
    # Extract the axes
    axes = np.array(figure.axes).reshape((n_dim, n_dim))

    # Loop over the histograms
    for yi in range(n_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            for i in range(n_plot_chains):
                ax.plot(chains[i, -n_last_spls:, xi], chains[i, -n_last_spls:, yi],
                        alpha=0.95, lw=0.75)
          
    plt.show(block=False)

    return figure

def draw_corner_nf(samples, true_val, labels, labelpad=10.0,
                frac_bounds=0.95):
    plt.rcParams.update({'font.size': 8})
    n_dim = samples.shape[-1]
    figure = corner.corner(samples,
                    labels=labels,
                    # var_names=var_names,
                    truths=true_val,
                    range=[frac_bounds] * n_dim,
                    labelpad=labelpad,
                    truth_color='g')
    figure.set_size_inches(7, 7)
    figure.suptitle('Visualize NF samples')
    return figure


def draw_kepler_results(chains, true_params, t, rv_obs, loss_vals, local_accs,
                        global_accs, rv_model, log_posterior, 
                        get_kepler_params_and_log_jac):

    n_chains = chains.shape[0]

    figure = plt.figure(figsize=(10,8))
    axs = [plt.subplot(221), plt.subplot(222), plt.subplot(223), plt.subplot(224)]
    plt.sca(axs[0])
    plt.plot(t, rv_obs, ".k", label='observations')
    x = np.linspace(0, 100, 500)
    plt.plot(x, rv_model(true_params, x), "C0", label='ground truth')

    chains_indx = np.random.choice(range(n_chains),
                                size=(np.minimum(n_chains,n_chains),),
                                replace=False)
    for id,i in enumerate(chains_indx):
        params, log_jac = get_kepler_params_and_log_jac(chains[i,-1,:])
        if id == 0:
            plt.plot(x, rv_model(params, x), c='gray', alpha=0.5, label='final samples')
        else:
            plt.plot(x, rv_model(params, x), c='gray', alpha=0.5)
    plt.xlabel('t')
    plt.ylabel('radial velocity')
    plt.legend()

    plt.sca(axs[1])
    posterior_evolution = jax.vmap(log_posterior)(chains[-10:,:,:].reshape(-1, 9)).reshape(10, -1)
    shift = max(jnp.max(posterior_evolution),0)
    plt.plot(- (posterior_evolution.T - shift))
    plt.yscale('log')
    plt.ylabel('walker negative log-likelihood')
    plt.xlabel('iteration')

    plt.sca(axs[2])
    plt.plot(loss_vals)
    plt.ylabel('Loss NF')
    plt.xlabel('iteration')
    plt.tight_layout()

    plt.sca(axs[3])
    plt.plot(local_accs.mean(0), label='local sampler')
    plt.plot(global_accs.mean(0), label='global sampler')
    plt.ylabel('Instantaneous acceptance rate')
    plt.xlabel('iteration')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    return figure