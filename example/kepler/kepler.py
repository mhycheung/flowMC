from re import S
import jax
import jax.numpy as jnp                # JAX NumPy
from jax.scipy.special import logsumexp
import numpy as np  
import optax 
from flax.training import train_state  # Useful dataclass to keep train state
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.MALA import mala_sampler
from nfsampler.sampler.NF_proposal import nf_metropolis_sampler
from nfsampler.nfmodel.utils import *

from utils import rv_model, log_likelihood, log_prior, sample_prior, get_kepler_params_and_log_jac

jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)

## Generate problem
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
prior_kwargs = {
    'ecc_alpha': 2, 'ecc_beta': 2,
    'log_k_mean': 1, 'log_k_var': 1,
    'v0_mean': 10, 'v0_var': 2,
    'log_period_mean': 2.5, 'log_period_var': 0.5,
    'log_s2_mean': -0.5, 'log_s2_var': 0.1,
}
# true_params = sample_prior(jax.random.PRNGKey(1), 1, **prior_kwargs)

random = np.random.default_rng(12345)
t = np.sort(random.uniform(0, 100, 50))
rv_err = 0.2
sigma2 = rv_err ** 2 + jnp.exp(2 * true_params[1])
rv_obs = rv_model(true_params, t) + random.normal(0, sigma2, len(t))

plt.plot(t, rv_obs, ".k")
x = np.linspace(0, 100, 500)
plt.plot(x, rv_model(true_params, x), "C0")
plt.show(block=False)



## Setting up sampling -- takes one input at the time.
def log_posterior(x):
    return  log_likelihood(x, t, rv_err, rv_obs) + log_prior(x, **prior_kwargs)

# log_posterior = jax.vmap(log_posterior_1)

d_log_posterior = jax.grad(log_posterior)

n_dim = 9
n_samples = 100
nf_samples = 100
n_chains = 100
n_layer = 3
n_hidden = 64
learning_rate = 0.01
momentum = 0.9
num_epochs = 5
batch_size = 100
n_iter = 10


print("Preparing RNG keys")
rng_key = jax.random.PRNGKey(42)
rng_key_init, rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,3)

rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)

print("Initializing chains.")
kepler_params_ini = sample_prior(rng_key_init, n_chains, **prior_kwargs)
# initial_position = kepler_params_ini
neg_logp_and_grad = jax.jit(jax.value_and_grad(lambda p: -log_posterior(p)))
optimized = []
for i in range(n_chains):
    soln = minimize(neg_logp_and_grad, kepler_params_ini.T[i], jac=True)
    optimized.append(jnp.asarray(get_kepler_params_and_log_jac(soln.x)[0]))

initial_position = jnp.stack(optimized).T

plt.figure()
plt.plot(t, rv_obs, ".k")
x = np.linspace(0, 100, 500)
plt.plot(x, rv_model(true_params, x), "C0")
for i in range(n_chains):
    params, log_jac = get_kepler_params_and_log_jac(kepler_params_ini[:, i])
    plt.plot(x, rv_model(params, x), c='gray', alpha=0.5)
for i in range(n_chains):
    params, log_jac = get_kepler_params_and_log_jac(optimized[i])
    plt.plot(x, rv_model(params, x), c='red', alpha=0.5)
plt.show(block=False)

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
    rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples,
                                                  log_posterior,
                                                  d_log_posterior,
                                                  initial_position, 0.001)
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



# Plot one chain 2 firsts coordinates to show the jump
plt.figure()
plt.plot(chains[0,:,0],chains[0,:,1])
plt.show(block=False)



value1 = true_params
# Make the base corner plot
figure = corner.corner(chains.reshape(-1,n_dim),
                labels=['v0', 'log_s2', 'log_period', 'log_k', 'sin_phi_', 'cos_phi_', 'ecc_', 'sin_w_', 'cos_w_'])
figure.set_size_inches(7, 7)

# Extract the axes
axes = np.array(figure.axes).reshape((n_dim, n_dim))
# Loop over the diagonal
for i in range(n_dim):
    ax = axes[i, i]
    ax.axvline(value1[i], color="g")
# Loop over the histograms
for yi in range(n_dim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(value1[xi], color="g")
        ax.axhline(value1[yi], color="g")
        ax.plot(value1[xi], value1[yi], "sg")
        ax.plot(chains[0, -1000:, xi],chains[0, -1000:, yi])

plt.show(block=False)

plt.figure()
plt.plot(t, rv_obs, ".k")
x = np.linspace(0, 100, 500)
plt.plot(x, rv_model(true_params, x), "C0")
for i in range(10):
    params, log_jac = get_kepler_params_and_log_jac(chains[i,-1,:])
    plt.plot(x, rv_model(params, x), c='gray', alpha=0.5)
plt.show(block=False)