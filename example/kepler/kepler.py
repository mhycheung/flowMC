import matplotlib as mpl
label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

import jax
import jax.numpy as jnp                # JAX NumPy
from jax.scipy.special import logsumexp
import numpy as np  
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import corner
import tqdm
import time
import pickle

from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.MALA import mala_sampler
from nfsampler.nfmodel.utils import *
from nfsampler.sampler.Sampler import Sampler
from nfsampler.utils.PRNG_keys import initialize_rng_keys

from utils import rv_model, log_likelihood, log_prior, sample_prior, get_kepler_params_and_log_jac
from plot import draw_corner, draw_kepler_results

jax.config.update("jax_enable_x64", True)

## Generate probelm
true_params = jnp.array([
    12.0, # v0
    np.log(0.5), # log_s2
    np.log(14.5), # log_P
    np.log(2.3), # log_k
    np.sin(1.5), # phi
    np.cos(1.5),
    0.4, # ecc
    np.sin(-0.7), # w
    np.cos(-0.7)
])
# prior_kwargs = { ## peaked
#     'ecc_alpha': 2, 'ecc_beta': 2,
#     'log_k_mean': 1, 'log_k_var': 1,
#     'v0_mean': 10, 'v0_var': 2,
#     'log_period_mean': 2.5, 'log_period_var': 0.5,
#     'log_s2_mean': -0.5, 'log_s2_var': 0.1,
# }
prior_kwargs = { ## flatter
    'ecc_alpha': 2, 'ecc_beta': 2,
    'log_k_mean': 1, 'log_k_var': 5,
    'v0_mean': 10, 'v0_var': 10,
    'log_period_mean': 1, 'log_period_var': 5,
    'log_s2_mean': -0.5, 'log_s2_var': 0.1,
}

n_obs = 17

random = np.random.default_rng(12345)
t = np.sort(random.uniform(0, 100, n_obs))
rv_err = 0.3
sigma2 = rv_err ** 2 + jnp.exp(2 * true_params[1])
rv_obs = rv_model(true_params, t) + random.normal(0, sigma2, len(t))
# plt.plot(t, rv_obs, ".k")
# x = np.linspace(0, 100, 500)
# plt.plot(x, rv_model(true_params, x), "C0")
# plt.show(block=False)

## Setting up sampling -- takes one input at the time.
def log_posterior(x):
    return  log_likelihood(x.T, t, rv_err, rv_obs) + log_prior(x,**prior_kwargs)
    
d_log_posterior = jax.grad(log_posterior)

config = {}
n_dim = 9
n_chains = 50

## long run
n_loop = 100
n_local_steps = 25
n_global_steps = 5
num_epochs = 10
## short run
# n_loop = 5
# n_local_steps = 10
# n_global_steps = 5
# num_epochs = 5

learning_rate = 0.01
momentum = 0.9
batch_size = n_chains
stepsize = 1e-5

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains,seed=42)

print("Initializing MCMC model and normalizing flow model.")

# initial_position = jax.random.normal(rng_key_set[0],shape=(n_chains,n_dim)) #(n_chains, n_dim)

kepler_params_ini = sample_prior(rng_key_set[0], n_chains,
                                 **prior_kwargs)
neg_logp_and_grad = jax.jit(jax.value_and_grad(lambda p: -log_posterior(p)))
optimized = []
for i in tqdm.tqdm(range(n_chains)):
    soln = minimize(neg_logp_and_grad, kepler_params_ini.T[i].T, jac=True)
    optimized.append(jnp.asarray(get_kepler_params_and_log_jac(soln.x)[0]))

initial_position = jnp.stack(optimized) #(n_chains, n_dim)
#planting initial position
print('planting initial position')
inital_position = initial_position.at[0,:].set(true_params)
# initial_position = kepler_params_ini.T

mean = initial_position.mean(0)
init_centered = (initial_position - mean)
cov = init_centered.T @ init_centered / n_chains

model = RealNVP(10, n_dim, 64, 1)

run_mala = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None),
                    out_axes=0)

print("Initializing sampler class")

nf_sampler = Sampler(n_dim, rng_key_set, model, run_mala,
                    log_posterior,
                    d_likelihood=d_log_posterior,
                    n_loop=n_loop,
                    n_local_steps=n_local_steps,
                    n_global_steps=n_global_steps,
                    n_chains=n_chains,
                    n_epochs=num_epochs,
                    n_nf_samples=100,
                    learning_rate=learning_rate,
                    momentum=momentum,
                    batch_size=batch_size,
                    stepsize=stepsize)


print("Sampling")

start = time.time()
_ = nf_sampler.sample(initial_position)
chains, nf_samples, local_accs, global_accs, loss_vals = _

print('Elapsed: ', time.time()-start, 's')

chains = np.array(chains)
nf_samples = np.array(nf_samples)

results = {
    'chains': chains,
    'nf_samples': nf_samples,
    'prior_samples': kepler_params_ini,
    'optimized_init': initial_position,
    'config': config,
    'true_params': true_params,
    'n_obs': n_obs,
    'rv_obs': rv_obs,
    't': t,
    'prior_kwargs': prior_kwargs,
    'rv_err': rv_err,
    'local_accs': local_accs,
    'global_accs': global_accs,
    'loss_vals': loss_vals
}

random_id = np.random.randint(10000)
with open('results_{:d}.pkl'.format(random_id), 'wb') as f:
    pickle.dump(results, f)
print("Saved with random id: {:d}".format(random_id))

print("Make plots")


# Make the base corner plot
labels = ['v0', 'log_s2', 'log_period', 'log_k', 'sin_phi_',
                            'cos_phi_', 'ecc_', 'sin_w_', 'cos_w_']
fig_corner = draw_corner(chains, true_params, labels, labelpad=0.3)

fig_results = draw_kepler_results(chains, true_params, t, rv_obs, loss_vals,
                                  rv_model, log_posterior, get_kepler_params_and_log_jac)




# value1 = true_params
# n_dim = n_dim
# # Make the base corner plot
# figure = corner.corner(chains.reshape(-1,n_dim),
#                 labels=['v0', 'log_s2', 'log_period', 'log_k', 'sin_phi_', 'cos_phi_', 'ecc_', 'sin_w_', 'cos_w_'], title_kwargs={"fontsize": 12})
# figure.set_size_inches(7, 7)
# figure.suptitle('Visualize initializations')
# # Extract the axes
# axes = np.array(figure.axes).reshape((n_dim, n_dim))
# # Loop over the diagonal
# for i in range(n_dim):
#     ax = axes[i, i]
#     ax.axvline(value1[i], color="g")
# # Loop over the histograms
# for yi in range(n_dim):
#     for xi in range(yi):
#         ax = axes[yi, xi]
#         ax.plot(value1[xi], value1[yi], "sg")
#         ax.plot(initial_position[:,xi], initial_position[:,yi],'+',ms=2)
#         ax.axvline(value1[xi], color="g")
#         ax.axhline(value1[yi], color="g")
# # plt.tight_layout()
# plt.show(block=False)



# with open('results_{:d}.pkl'.format(random_id), 'rb') as f:
#     results_ = pickle.load(f)