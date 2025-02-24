import numpy as np
from tqdm import tqdm

from .utilities.distributions import *

# for use with MLE
import optimagic as om

def ndarray_to_list_of_dicts(x, key_map):
    return {k: x[v] for k, v in key_map.items()}

class Sampler:
    def __init__(self, density_model, priors):
        self._samples = None
        self._acceptances = None
        self._lp = None

        self.priors = priors
        self.model = density_model

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples


    @property
    def acceptances(self):
        return self._acceptances

    @acceptances.setter
    def acceptances(self, acceptances):
        self._acceptances = acceptances


    @property
    def lp(self):
        return self._lp

    @lp.setter
    def lp(self, lp):
        self._lp = lp


    def log_prior(self, theta):
        return self.priors.logpdf(theta)

    def log_posterior(self, theta, **kwargs):
        logprior = self.log_prior(theta)
        if not np.isfinite(logprior):
            return -np.inf

        return logprior + self.model.log_likelihood(theta, **kwargs)


    def initialize(self, init_theta, chain_len, **kwargs):
        # preallocate chain
        self.samples = np.zeros((chain_len, len(init_theta)))
        self.acceptances = np.zeros(chain_len)
        self.lp = np.zeros(chain_len)

    def iterate(self, theta, iter, **kwargs):
        raise NotImplementedError("iteration not defined for this class of algorithms")

    def sample(self, chain_len, **kwargs):
        # draw from the prior
        theta = self.priors.rand()
        self.initialize(theta, chain_len)

        for iter in tqdm(range(1, chain_len)):
            # MCMC step
            theta, lp, a = self.iterate(theta, **kwargs)

            # update chain
            self.samples[iter] = list(theta.values())
            self.lp[iter] = lp
            self.acceptances[iter] = a

        return self.samples

    def acceptance_ratio(self):
        return np.mean(self.acceptances)

class MetropolisHastings(Sampler):
    def __init__(self, density_model, priors, step_size=0.01):
        super().__init__(density_model, priors)
        
        # only stick with Gaussian proposals for now
        self.step_size = step_size

    def propose(self, theta, **kwargs):
        return Prior(
            {k: Normal(v, self.step_size) for k,v in theta.items()}
        ).rand()

    # add optional argument for auto tuning
    def iterate(self, theta, **kwargs):
        theta_proposed = self.propose(theta, **kwargs)

        logprob      = self.log_posterior(theta_proposed, **kwargs)
        prev_logprob = self.log_posterior(theta, **kwargs)
        alpha = min(1, np.exp(logprob-prev_logprob))

        if np.random.rand() <= alpha:
            return theta_proposed, logprob, 1
        else:
            return theta, prev_logprob, 0

class Slice(Sampler):
    def __init__(self, density_model, priors, w, max_iters=20):
        super().__init__(density_model, priors)

        # set the step size for all parameters
        self.w = {k: w for k in priors.dists.keys()}
        self.max_iters = max_iters

    def iterate(self, theta, **kwargs):
        theta0 = theta.copy()
        nstep_out = nstep_in = 0

        # define boundaries
        thetal = theta0.copy()
        thetar = theta0.copy()

        for i, wi in self.w.items():
            # uniformly sample from 0 to p(theta) in log space
            logprob = self.log_posterior(theta, **kwargs) - np.random.standard_exponential()

            # create initial interval
            thetal[i] = theta[i] - np.random.uniform() * wi
            thetar[i] = thetal[i] + wi

            # stepping out procedure
            iter = 0
            while logprob <= self.log_posterior(thetal, **kwargs):
                thetal[i] -= wi
                iter += 1
                # if iter > self.max_iters:
                #     raise RuntimeError("exceeded max iters")
            nstep_out += iter

            iter = 0
            while logprob <= self.log_posterior(thetar, **kwargs):
                thetar[i] += wi
                iter += 1
                # if iter > self.max_iters:
                #     raise RuntimeError("exceeded max iters")
            nstep_out += iter

            iter = 0
            theta[i] = np.random.uniform(thetal[i], thetar[i])

            while logprob > self.log_posterior(theta, **kwargs):
                if theta[i] > theta0[i]:
                    thetar[i] = theta[i]
                elif theta[i] < theta0[i]:
                    thetal[i] = theta[i]
                theta[i] = np.random.uniform(thetal[i], thetar[i])
                iter += 1
                # if iter > self.max_iters:
                #     raise RuntimeError("exceeded max iters")
            nstep_in += iter

            # reset bounds to the accepted points
            thetar[i] = thetal[i] = theta[i]

        return theta, logprob, 1

class MaximumLikelihood:
    def __init__(self, density_model, priors):
        self.priors = priors
        self.model = density_model

    # just perform constrained optimization for now
    def optimize(self, algo, bounded=False, **kwargs):
        bounds = om.Bounds(
            lower = {k: v[0] for k,v in self.priors.support.items()},
            upper = {k: v[1] for k,v in self.priors.support.items()}
        )
        return om.maximize(
            lambda theta: self.model.log_likelihood(theta),
            self.priors.rand(),
            bounds = bounds if bounded else None,
            algorithm = algo,
            **kwargs
        )