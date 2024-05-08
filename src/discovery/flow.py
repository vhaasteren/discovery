import time
import collections.abc
import typing
PyTree = typing.Any

import numpy as np
import matplotlib.pyplot as pp
import tqdm

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import flowjax.distributions


# modified from flowjax/train/losses.py

class ElboLoss:
    """Negative evidence lower bound (ELBO), approximated using samples.

    Args:
        target:      Target log posterior density up to an additive constant.
        num_samples: Minibatch size.
    """

    target: collections.abc.Callable[[jax.typing.ArrayLike], jax.Array]
    num_samples: int

    def __init__(self,
                 target: collections.abc.Callable[[jax.typing.ArrayLike], jax.Array],
                 num_samples: int):
        self.target = target
        self.num_samples = num_samples

    @eqx.filter_jit
    def __call__(self,
                 params: flowjax.distributions.AbstractDistribution,
                 static: flowjax.distributions.AbstractDistribution,
                 key: jax.Array,
                 beta: float = 1.0):
        """Compute the ELBO loss.

        Args:
            params: the trainable parameters of the model.
            static: the static components of the model.
            key:    JAX random seed.
            beta:   inverse temperature applied to target likelihood.
        """
        dist = eqx.combine(params, static)

        # requires only forward pass through the flow.
        samples, log_probs = dist.sample_and_log_prob(key, (self.num_samples,))
        target_density = jax.vmap(self.target)(samples)
        return (log_probs - beta * target_density).mean()


# modified from flowjax/train/train_utils.py to take combined loss + gradient,
# and to average over repeated minibatches

def step(params: PyTree,
         static: PyTree,
         key: jax.Array,
         beta: float,
         optimizer: optax.GradientTransformation,
         opt_state: PyTree,
         gradloss_fn: collections.abc.Callable,
         multibatch: int):
    """Carry out a training step.

    Args:
        params:      parameters for the model
        static:      static components of the model
        key:         JAX key passed to loss function
        beta:        inverse temperature passed to loss function
        optimizer:   Optax optimizer
        opt_state:   optimizer state
        gradloss_fn: loss and its gradient, e.g., from eqx.filter_value_and_grad(loss_fn)
        multibatch:  number of minibatches in this batch

    Returns:
        tuple:       (params, opt_state, loss_val)
    """
    losses, grads = [], []
    for i in range(multibatch):
        key, subkey = jax.random.split(key)
        loss, grad = gradloss_fn(params, static, subkey, beta)
        losses.append(loss)
        grads.append(grad)

    loss = np.mean(losses)
    grads = jax.tree_util.tree_map(lambda *x: sum(x) / multibatch, *grads)

    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss


# modified from flowjax/train/variational_fit.py

class VariationalFit:
    def __init__(self, dist, loss_fn, multibatch=1, optimizer=None, learning_rate=5e-4, annealing_schedule=None, patience=100, show_progress=True):
        self.dist, self.loss_fn = dist, loss_fn
        self.gradloss_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))
        self.multibatch = multibatch

        self.optimizer = optimizer or optax.adam(learning_rate)
        params, static = eqx.partition(self.dist, eqx.is_inexact_array, is_leaf=lambda leaf: isinstance(leaf, flowjax.wrappers.NonTrainable))
        self.opt_state = self.optimizer.init(params)
        self.annealing_schedule = annealing_schedule or (lambda iter: 1.0)
        self.patience = patience

        self.show_progress = show_progress

        self.iter = 0
        self.losses = []
        self.best_params, self.best_iter = None, 0

    def run(self, key, steps = 100):
        params, static = eqx.partition(
            self.dist,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, flowjax.wrappers.NonTrainable),
        )

        if self.best_params:
            params = self.best_params

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, steps)

        if self.show_progress == True:
            keys = tqdm.tqdm(keys)

        for key in keys:
            params, self.opt_state, loss = step(params, static, key, beta=jnp.asarray(self.annealing_schedule(self.iter)),
                                                optimizer=self.optimizer, opt_state=self.opt_state,
                                                gradloss_fn=self.gradloss_fn, multibatch=self.multibatch)

            self.losses.append(loss.item())

            if loss.item() == min(self.losses):
                self.best_params = params
                self.best_iter = self.iter

            if self.iter - self.best_iter > self.patience:
                print(f'Early stopping at iteration {self.iter}')
                break

            if self.show_progress == True:
                keys.set_postfix({"loss": f'{self.losses[-1]:.2f}'})
            elif callable(self.show_progress):
                self.show_progress(eqx.combine(self.best_params, static), self.losses[-1], self.annealing_schedule(self.iter))

            self.iter = self.iter + 1

        if callable(self.show_progress) and hasattr(self.show_progress, 'freeze'):
            self.show_progress.freeze()

        return key, eqx.combine(self.best_params, static)


# obsolete, use VariationalFit class

def fit_to_variational_target(
    key: jax.Array,
    dist: flowjax.distributions.AbstractDistribution,
    loss_fn: collections.abc.Callable,
    steps: int = 100,
    startiter: int = 0,
    multibatch: int = 1,
    optimizer: optax.GradientTransformation | None = None,
    opt_state: optax.OptState | None = None,
    learning_rate: float = 5e-4,
    annealing_schedule: collections.abc.Callable | None = None,
    return_best: bool = True,
    patience: int = 100,
    show_progress: collections.abc.Callable | bool = True,
) -> tuple[flowjax.distributions.AbstractDistribution, list]:
    """Train a distribution (e.g., a flow) by variational inference.

    Args:
        key:                JAX PRNGKey
        dist:               flowjax distribution object
        loss_fn:            loss function
        steps:              number of training steps
        startiter:          starting iteration number
        multibatch:         number of minibatches per gradient
        optimizer:          Optax optimizer
        opt_state:          last state of optimizer
        learning_rate:      learning rate of default Adam optimizer
        annealing_schedule: annealing schedule function beta(iter)
        return_best:        return minimum-loss parameters
        patience:           patience for early stopping
        show_progress:      if True, show progress bar; if callable, call
                            show_progress(best_flow, last_loss, beta)

    Returns:
        tuple:              (flow, losses, optimizer, opt_state)
    """

    params, static = eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, flowjax.wrappers.NonTrainable),
    )

    if opt_state is None:
        if optimizer is None:
            optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

    losses = []

    best_params, best_iter = params, 0
    keys = jax.random.split(key, steps)

    if show_progress == True:
        keys = tqdm.tqdm(keys)

    gradloss_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))

    iter = startiter
    for key in keys:                 # was flowjax.train.train_utils.step
        params, opt_state, loss = step(
            params,
            static,
            key,
            jnp.asarray(1.0 if annealing_schedule is None else annealing_schedule(iter)),
            optimizer=optimizer,
            opt_state=opt_state,
            gradloss_fn=gradloss_fn, # was loss_fn=loss_fn
            multibatch=multibatch,   # was nothing
        )

        losses.append(loss.item())

        if loss.item() == min(losses):
            best_params = params
            best_iter = iter

        if iter - best_iter > patience:
            print(f'Early stopping at iteration {iter}')
            break

        if show_progress == True:
            keys.set_postfix({"loss": f'{losses[-1]:.2f}'})
        elif callable(show_progress):
            show_progress(eqx.combine(best_params, static), losses[-1],
                          1.0 if annealing_schedule is None else annealing_schedule(iter))

        iter = iter + 1

    if callable(show_progress) and hasattr(show_progress, 'freeze'):
        show_progress.freeze()

    return eqx.combine(best_params if return_best else params, static), losses, optimizer, opt_state


def hellinger(c1, c2, bins=100, range=None):
    """Hellinger distance of two vectors of variates."""

    if range is None:
        range = (np.min(c1), np.max(c1))

    h1, bn = np.histogram(c1, bins=bins, range=range, density=True)
    h2, _  = np.histogram(c2, bins=bins, range=range, density=True)

    h1 = h1 / np.sum(h1)
    h2 = h2 / np.sum(h2)

    return 1.0 - np.sum(np.sqrt(h1 * h2))


import IPython.display

class display_flow:
  def __init__(self, logx, cadence=50, refchain=None, vlogx=None, save=None):
      self.logx = logx
      self.cadence, self.refchain, self.vlogx, self.save = cadence, refchain, vlogx, save

      self.t0, self.frozen = time.time(), None
      self.losses = []

  def freeze(self):
      self.frozen = time.time() - self.t0

  def thaw(self):
      if self.frozen is not None:
        self.t0 = time.time() - self.frozen
        self.frozen = None

  def __call__(self, flow, loss, temp):
      self.thaw()
      self.losses.append(loss)
      iteration = len(self.losses) - 1

      if callable(self.cadence):
        if not self.cadence(iteration):
          return
      else:
        if iteration % self.cadence != 0:
            return

      # ps = logx.to_df(flow.sample(train_key, sample_shape=(2*4096,)))

      train_key = jax.random.PRNGKey(42)

      if self.vlogx is None:
        ps = self.logx.to_df(flow.sample(train_key, sample_shape=(2*4096,)))
      else:
        samples, log_probs = flow.sample_and_log_prob(train_key, (4096,))
        ps = self.logx.to_df(samples)
        logweights = np.concatenate([self.vlogx(samples[i*512:(i+1)*512,:]) - log_probs[i*512:(i+1)*512]
                                     for i in range(8)])
        weights = np.exp(logweights - np.max(logweights))

      if callable(self.save):
          self.save(iteration, ps)

      p1, p2 = self.logx.params[-2:]
      h1, h2 = 0.0, 0.0

      pp.clf()
      pp.gcf().set_figwidth(10); pp.gcf().set_figheight(2.5)
      try:
        pp.subplot(1,4,1); pp.plot(np.array(self.losses[-200:]) - self.losses[-1]); pp.xlabel('iter (last 200)'); pp.ylabel('loss')

        pp.subplot(1,4,2); pp.plot(ps[p1], ps[p2], '.', ms=1.0); pp.xlabel(p1); pp.ylabel(p2)

        pp.subplot(1,4,3); pp.hist(np.array(ps[p1]), histtype='step', density=True, bins=40); pp.xlabel(p1)

        if self.refchain is not None:
            pp.hist(np.array(self.refchain[p1]), histtype='step', density=True, bins=40)
            h1 = hellinger(self.refchain[p1], ps[p1])

        if self.vlogx is not None:
            pp.hist(ps[p1], histtype='step', density=True, bins=40, weights=weights)

        pp.subplot(1,4,4); pp.hist(np.array(ps[p2]), histtype='step', density=True, bins=40, label='nf'); pp.xlabel(p2);

        if self.refchain is not None:
            pp.hist(np.array(self.refchain[p2]), histtype='step', density=True, bins=40, label='ref')
            h2 = hellinger(self.refchain[p2], ps[p2])

        if self.vlogx is not None:
            pp.hist(ps[p2], histtype='step', density=True, bins=40, weights=weights, label='rw')

        pp.legend()
      except:
        pass

      pp.suptitle(f'Iteration {iteration}, time = {(time.time() - self.t0):.0f}, loss = {self.losses[-1]:.2f}, beta = {temp:.2f}, hdist = {h1:.3f}, {h2:.3f}')
      pp.tight_layout();

      IPython.display.display(pp.gcf())
      IPython.display.clear_output(wait=True)
