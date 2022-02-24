from re import S
import jax
import jax.numpy as jnp                # JAX NumPy
from jax.scipy.special import logsumexp
import numpy as np  
import optax 
from flax.training import train_state  # Useful dataclass to keep train state
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.MALA import mala_sampler
from nfsampler.sampler.NF_proposal import nf_metropolis_sampler
from nfsampler.nfmodel.utils import *

from utils import rv_model, log_likelihood, log_prior, sample_prior

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

random = np.random.default_rng(12345)
t = np.sort(random.uniform(0, 100, 50))
rv_err = 0.3
rv_obs = rv_model(true_params, t) + random.normal(0, rv_err, len(t))

plt.plot(t, rv_obs, ".k")
x = np.linspace(0, 100, 500)
plt.plot(x, rv_model(true_params, x), "C0")
plt.show(block=False)

prior_kwargs = {
    'ecc_alpha': 2, 'ecc_beta': 2,
    'log_k_mean': 1, 'log_k_var': 1,
    'v0_mean': 10, 'v0_var': 2,
    'log_period_mean': 2, 'log_period_var': 1,
    'log_s2_mean': 0, 'log_s2_var': 1,
}


## Setting up sampling -- takes one input at the time.
def log_posterior(x):
    return  log_likelihood(x, t, rv_err, rv_obs) + log_prior(x, **prior_kwargs)

# log_posterior = jax.vmap(log_posterior_1)

d_log_posterior = jax.grad(log_posterior)

n_dim = 9
n_samples = 20
nf_samples = 100
n_chains = 100
n_layer = 10
n_hidden = 64
learning_rate = 0.01
momentum = 0.9
num_epochs = 3
batch_size = 100
n_iter = 10


print("Preparing RNG keys")
rng_key = jax.random.PRNGKey(42)
rng_key_init, rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,3)

rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)

print("Initializing chains.")
## Dummy intialization for now - only one point
# neg_logp_and_grad = jax.jit(jax.value_and_grad(lambda p: -log_likelihood(p, t, rv_err, rv_obs)))
# soln = minimize(neg_logp_and_grad, true_params, jac=True)
# kepler_params_ini = jnp.asarray(get_kepler_params_and_log_jac(soln.x)[0])
# kepler_params_ini = kepler_params_ini.reshape(1, -1).repeat(n_chains, 0)
# initial_position = kepler_params_ini.T

kepler_params_ini = sample_prior(rng_key_init, n_chains, **prior_kwargs)
initial_position = kepler_params_ini

print("Initializing normalizing flow model.")

 ## not ML standard

model = RealNVP(n_layer, n_dim, n_hidden, dt=1)
params = model.init(init_rng_keys_nf, jnp.ones((batch_size,n_dim)))['params']

run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 1, None),
                    out_axes=0)

tx = optax.adam(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
 
print("Sampling")

def sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, initial_position):
    #rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, log_posterior, initial_position)
    rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples,
                                                  log_posterior,
                                                  d_log_posterior,
                                                  initial_position, 0.01)
    flat_chain = positions.reshape(-1,n_dim)
    rng_keys_nf, state = train_flow(rng_key_nf, model, state, flat_chain, 
                                    num_epochs, batch_size)
    rng_keys_nf, nf_chain, log_prob, log_prob_nf = nf_metropolis_sampler(
        rng_keys_nf, nf_samples, model, state.params, jax.vmap(log_posterior),
        positions[:,-1]
        )

    positions = jnp.concatenate((positions,nf_chain),axis=1)
    return rng_keys_nf, rng_keys_mcmc, state, positions

last_step = initial_position
chains = []
for i in range(n_iter):
	rng_keys_nf, rng_keys_mcmc, state, positions = sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, last_step)
	last_step = positions[:,-1].T
	chains.append(positions)
chains = np.concatenate(chains,axis=1)
nf_samples = sample_nf(model, state.params, rng_keys_nf, 10000)

import corner
# import matplotlib.pyplot as plt

# Plot one chain 2 firsts coordinates to show the jump
plt.figure()
plt.plot(chains[0,:,0],chains[0,:,1])
plt.show(block=False)


# Plot all chains
plt.figure()
corner.corner(chains.reshape(-1,n_dim),
                labels=['v0', 'log_s2', 'log_period', 'log_k', 'sin_phi_', 'cos_phi_', 'ecc_', 'sin_w_', 'cos_w_'])

## Example code from corner to overplot - put true params
# # This is the true mean of the second mode that we used above:
# value1 = mean
# # This is the empirical mean of the sample:
# value2 = np.mean(samples, axis=0)
# # Make the base corner plot
# figure = corner.corner(samples)
# # Extract the axes
# axes = np.array(figure.axes).reshape((ndim, ndim))
# # Loop over the diagonal
# for i in range(ndim):
#     ax = axes[i, i]
#     ax.axvline(value1[i], color="g")
#     ax.axvline(value2[i], color="r")
# # Loop over the histograms
# for yi in range(ndim):
#     for xi in range(yi):
#         ax = axes[yi, xi]
#         ax.axvline(value1[xi], color="g")
#         ax.axvline(value2[xi], color="r")
#         ax.axhline(value1[yi], color="g")
#         ax.axhline(value2[yi], color="r")
#         ax.plot(value1[xi], value1[yi], "sg")
#         ax.plot(value2[xi], value2[yi], "sr")

plt.show(block=False)
