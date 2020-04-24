import gpflow
import tensorflow as tf
import numpy as np

def probit(x):
    return 0.5*(1.0+tf.erf(x/np.sqrt(2.0))) * (1-2e-3) + 1e-3


class BetaBetaMix(gpflow.likelihoods.Likelihood):
    def __init__(self, invlink=probit, outlier_prob=0.01, scale=1.0, a_prime=10., b_prime=1.):
        gpflow.likelihoods.Likelihood.__init__(self)
        self.scale = gpflow.params.Parameter(scale, gpflow.transforms.positive)
        self.outlier_prob = gpflow.params.Parameter(outlier_prob)
        self.invlink = invlink
        self.a_prime, self.b_prime = np.array(a_prime), np.array(b_prime)

    def logp(self, F, Y):
        mean = self.invlink(F) 
        alpha = mean * self.scale
        beta = self.scale - alpha
        logp_beta = gpflow.logdensities.beta(Y, alpha, beta)
        logp_beta_prime = gpflow.logdensities.beta(Y, self.a_prime, self.b_prime)
        return tf.log(self.outlier_prob * tf.exp(logp_beta_prime) + (1. - self.outlier_prob) * tf.exp(logp_beta))

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        mean = self.invlink(F)
        return (mean - tf.square(mean)) / (self.scale + 1.)
