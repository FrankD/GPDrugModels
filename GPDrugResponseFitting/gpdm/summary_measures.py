import numpy as np
from scipy.stats import beta
from .models import norm_cdf


# Functions to calculate summary measures on GP fits of dose-response
# curves

# Calculate area under the curve from GP mean and variance
def get_AUC(m_vb, dosages=None):
    if dosages is None:
        xx = np.linspace(1e-6, 1, 100).reshape(-1, 1)
    else:
        xx = dosages

    mu, var = m_vb.predict_f(xx)
    
    vb_means, vb_uppers, vb_lowers, lstsqs = [], [], [], []
    
    for i, (mu, var) in enumerate(zip(mu.T, var.T)):
        vb_means.append(norm_cdf(mu).mean())
        vb_lowers.append(norm_cdf(mu - 2*np.sqrt(var)).mean())
        vb_uppers.append(norm_cdf(mu + 2*np.sqrt(var)).mean())

    return [vb_means, vb_uppers, vb_lowers]


# Calculate area under the curve (by sampling)
def get_AUC_samples(m_vb, num_samples=100, dosages=None):
    if dosages is None:
        xx = np.linspace(1e-6, 1, 100).reshape(-1, 1)
    else:
        xx = dosages

    # samples
    samples = m_vb.predict_f_samples(xx, num_samples)
    samples = norm_cdf(samples)
    
    aucs = samples.mean(axis=1)
    
    return aucs

def get_AUC_stats(m_vb, num_samples=100, dosages=None):
    
    auc_samples = get_AUC_samples(m_vb, num_samples=num_samples, dosages=dosages)

    return np.mean(auc_samples), np.std(auc_samples)


def get_AUC_probs(m_vb, num_samples=100, resist_mean=0.9, dosages=None):
    # Calculate alpha such that mean is resist_mean, and beta=1:
    # (Note: sensitive_mean = 1-resist_mean, with alpha and beta swapped)
    alpha = resist_mean / (1-resist_mean)

    auc_samples = get_AUC_samples(m_vb, num_samples=num_samples, dosages=dosages)
    print(np.mean(auc_samples))

    auc_probs_resistant = beta.pdf(auc_samples, alpha, 1)
    
    auc_probs_sensitive = beta.pdf(auc_samples, 1, alpha)

    
    print(np.log(sum(auc_probs_resistant)))
    print(np.log(sum(auc_probs_sensitive)))
    
    #if(np.mean(auc_samples) < 0.4):
    #    raise 1

    # sum over samples to integrate over GP draws
    return np.log(sum(auc_probs_resistant)), np.log(sum(auc_probs_sensitive))

def get_IC50_stats(m_vb, num_samples=100):
    
    temp, ic50_samples = get_ic50_samples(m_vb, num_samples=num_samples)

    return np.mean(ic50_samples), np.std(ic50_samples)


# Calculate IC50 point (by sampling)
def get_ic50_samples(m_vb, num_samples=500,
             extrapolate_ic50=True):

    if(extrapolate_ic50):
        xx = np.logspace(np.log10(0.001), np.log10(100), num=1000)[:, None]
    else:
        xx = np.linspace(0, 1, 100)[:, None]

    # get the IC50
    samples = m_vb.predict_f_samples(xx, num_samples)
    samples = norm_cdf(samples)
        
    sample_crosses_50 = np.any(samples < 0.5, axis=1)
    crossing_index = np.argmax(samples < 0.5, 1)

    # work out whether the sample is increasing:
    sample_is_increasing = []
    
    for n in range(samples.shape[2]):  # loop over outputs
        i = crossing_index[:, n]
        i_plus_1 = np.fmin(i+1, samples.shape[1]-1)
        diff = samples[np.arange(samples.shape[0]), i_plus_1, n] - samples[np.arange(samples.shape[0]), i, n]
        sample_is_increasing.append(np.where(diff >= 0, True, False))
    sample_is_increasing = np.array(sample_is_increasing).T

    IC50 = xx.flatten()[crossing_index]
    IC50 = np.where(sample_crosses_50, IC50, np.nan)
    IC50 = np.where(sample_is_increasing, np.nan, IC50)

    p_sample_crosses = 1.-np.mean(np.isnan(IC50), 0)
    IC50 = [ic[~np.isnan(ic)] for ic in IC50.T]
    return p_sample_crosses, IC50


def ic50_summary(ic50_samples, max_dosage, dilution=8):
    ic50_samples = (1-np.array(ic50_samples))*8*np.log(2) + np.log(max_dosage)  # Same scale as Sanger 

    ic50_means = [np.mean(e) for e in ic50_samples]
    ic50_lower = [np.percentile(s, 5) if len(s) > 2 else np.nan for s in ic50_samples]
    ic50_upper = [np.percentile(s, 95) if len(s) > 2 else np.nan for s in ic50_samples]
    return ic50_means, ic50_upper, ic50_lower