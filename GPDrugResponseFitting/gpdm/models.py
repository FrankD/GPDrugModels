# coding: utf-8

import gpflow
import numpy as np
import pandas as pd
from .mixed_likelihood import BetaBetaMix
import tensorflow as tf
from scipy.special import erf
from matplotlib import pyplot as plt

def norm_cdf(x):
    return 0.5*(1.0+erf(x/np.sqrt(2.0))) * (1-2e-3) + 1e-3

def sigmoid(x):
    return 1./(1.+tf.exp(-x))


class Lstsq_sigmoid(gpflow.models.Model):
    def __init__(self, X, Y):
        gpflow.models.Model.__init__(self)
        self.X, self.Y = X, Y
        self.W = gpflow.param.Param(np.zeros((self.X.shape[1], self.Y.shape[1])))
        self.b = gpflow.param.Param(np.zeros(self.Y.shape[1]))

    def build_likelihood(self):
        f = tf.matmul(tf.log(self.X), self.W) + self.b
        y = sigmoid(f)
        err = tf.reduce_sum(tf.square(y - self.Y))
        return -err

    @gpflow.autoflow((tf.float64,))
    def predict(self, X):
        f = tf.matmul(tf.log(X), self.W) + self.b
        return sigmoid(f)

    def get_ic50(self):
        return np.exp(-self.b.value.squeeze() / self.W.value.squeeze())


def _plot_responder(m, ax=None):

    # calc plotting grid
    xmin, xmax = m.X.value.min(), m.X.value.max()
    xmin, xmax = xmin - 0.1*(xmax - xmin), xmax + .1*(xmax-xmin)
    Xtest = np.linspace(xmin, xmax, 200)

    #predict, get quantiles
    def probit(x):
        return 0.5*(1.0 + erf(x/np.sqrt(2.0))) * (1-2e-3) + 1e-3
    mu, var = m.predict_f(Xtest.reshape(-1, 1))
    mean, upper, lower = probit(mu), probit(mu + 2*np.sqrt(var)), probit(mu - 2*np.sqrt(var))

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(m.X.value[1:, 0], m.Y.value[1:, 0], 'kx', mew=1.5)  # data
    ax.plot(m.X.value[0, 0], 1.0, 'rx', mew=1.5)  # control point
    ax.plot(Xtest, mean, 'r', lw=1.6)
    ax.plot(Xtest, lower, 'r--', lw=1)
    ax.plot(Xtest, upper, 'r--', lw=1)
    ax.set_ylim(0, 1.2)
    return ax

def _plot_nonresponder(m, ax=None):

    # calc plotting grid
    xmin, xmax = m.X.value.min(), m.X.value.max()
    xmin, xmax = xmin - 0.1*(xmax - xmin), xmax + .1*(xmax-xmin)
    Xtest = np.linspace(xmin, xmax, 200)

    #predict, get quantiles
    def probit(x):
        return 0.5*(1.0 + erf(x/np.sqrt(2.0))) * (1-2e-3) + 1e-3
    mu, var = m.predict_f(Xtest.reshape(-1, 1))
    mean, upper, lower = probit(mu), probit(mu + 2*np.sqrt(var)), probit(mu - 2*np.sqrt(var))

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(m.X.value[:, 0], m.Y.value[:, 0], 'kx', mew=1.5)  # data
    ax.plot(Xtest, mean, 'b', lw=1.6)
    ax.plot(Xtest, lower, 'b--', lw=1)
    ax.plot(Xtest, upper, 'b--', lw=1)
    ax.set_ylim(0, 1.2)
    return ax




def responder_model(X, Y, control_dosage=None):
    """
    A GP model of the dosage-response curve, designed to fit well to responding experiments. 

    X is a np.array of log-dosages

    Y is a np.array of responses, normalized to [0, 1]

    control_dosage is an 'extra' X value at which we assume no effect(i.e.
    Y=1). By default this is taken to be smaller than the smallest value in X,
    by a distance given by the average gap in X.
    """

    # ensure X and Y are correctly shape for gpflow: should have one column each.
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)

    # find default control dosage if needed
    if control_dosage is None:
        control_dosage = np.min(X.flatten()) - np.mean(np.diff(np.sort(X.flatten())))

    k = gpflow.kernels.Matern32(1) + gpflow.kernels.Linear(1)

    # put in extra observation to X and Y
    X = np.vstack((np.array([[control_dosage]]), X))
    Y = np.vstack((np.array([1, 0]), np.hstack([Y, np.ones(Y.shape)])))


    # quite a complex likelihood! The first datum is the control, it is observed
    # with beta noise, small width. The remaining data are observed with a robust noise
    # model given by a mixture of betas.
    lik0 = gpflow.likelihoods.Beta()
    #lik1 = gpflow.likelihoods.Beta()
    #lik0 = BetaBetaMix(outlier_prob=0.001)
    lik1 = BetaBetaMix(outlier_prob=0.001)
    lik0.scale = 50
    lik1.scale = 50
    lik = gpflow.likelihoods.SwitchedLikelihood([lik0, lik1])

    #meanf = gpflow.mean_functions.Constant(3)

    m = gpflow.models.VGP(X=X, Y=Y, kern=k, #mean_function=meanf,
                          likelihood=lik, num_latent=1)

    # set and fix sensible parameters for the kernel
    m.kern.kernels[0].variance = .2
    m.kern.kernels[0].lengthscales = 8.
    m.kern.kernels[1].variance = .1
    m.kern.trainable = False
    m.likelihood.trainable = False
    m.plot = lambda :_plot_responder(m)

    return m


def nonresponder_model(X, Y, control_dosage=None):
    """
    A GP model of the dosage-response curve, designed to fit well to NON-responding experiments.

    X is a np.array of log-dosages

    Y is a np.array of responses, normalized to [0, 1]
    """

    # ensure X and Y are correctly shape for gpflow: should have one column each.
    X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)

    # find default control dosage if needed
    if control_dosage is None:
        control_dosage = np.min(X.flatten()) - np.mean(np.diff(np.sort(X.flatten())))
    k = gpflow.kernels.Constant(1)

    # put in extra observation to X and Y
    X = np.vstack((np.array([[control_dosage]]), X))
    Y = np.vstack((np.array([1, 0]), np.hstack([Y, np.ones(Y.shape)])))

    # quite a complex likelihood! The first datum is the control, it is observed
    # with beta noise, small width. The remaining data are observed with a robust noise
    # model given by a mixture of betas.
    lik0 = gpflow.likelihoods.Beta()
    lik1 = BetaBetaMix(outlier_prob=0.001)
    lik0.scale = 50.
    lik1.scale = 50.
    lik = gpflow.likelihoods.SwitchedLikelihood([lik0, lik1])

    m = gpflow.vgp.VGP(X=X, Y=Y, kern=k,
                          likelihood=lik, num_latent=1)

    # set and fix sensible parameters for the kernel
    m.kern.variance = .3
    m.kern.fixed = True
    m.likelihood.fixed = True

    m.plot = lambda :_plot_nonresponder(m)

    return m
