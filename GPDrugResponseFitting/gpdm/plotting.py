import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = (12, 7)
from matplotlib import pyplot as plt
plt.style.use('seaborn-talk')
import numpy as np
from .models import norm_cdf

def get_plot_data(m_vb, extrapolate_ic50=True):
    if(extrapolate_ic50):
        xx = np.linspace(0, 2, 200).reshape(-1, 1)
        plot_lim = 2
    else:
        xx = np.linspace(0, 1, 100).reshape(-1, 1)
        plot_lim = 1

    m_vb._needs_recompile = True  # work around GPflow bug
    mu, var = m_vb.predict_f(xx)

    mean = norm_cdf(mu)
    lower = norm_cdf(mu - 2*np.sqrt(var))
    upper = norm_cdf(mu + 2*np.sqrt(var))

    return xx, mean, lower, upper, plot_lim



# Plotting with IC50 histogram and max concentration
def plot(m_vb, m_lstsq=None, ic50_samples=[], extrapolate_ic50=True,
    ylim=None, labelsize=12, lw_scale=1):

    f, ax = plt.subplots(1, 1)

    xx, mean, lower, upper, plot_lim = get_plot_data(m_vb, extrapolate_ic50)

    meanplot, = ax.plot(xx, mean, lw=2*lw_scale)  # , label=legend)
    c = meanplot.get_color()
    ax.plot(xx, lower, '--', color=c, lw=1*lw_scale)
    ax.plot(xx, upper, '--', color=c, lw=1*lw_scale)

    ax.plot(m_vb.X.value, m_vb.Y.value[:,0], 'o', color=c)

    if(m_lstsq!=None):
      fit = m_lstsq.predict(xx)
      ax.plot(xx, fit, 'b')
      ax.plot([m_lstsq.get_ic50()]*2, [0, 0.5], 'b--')
    
    if(extrapolate_ic50):
        ax.plot([1,1], [0, 1], 'g--')
    
    if not(ylim is None):
        ax.set_ylim([0, ylim])
    else:
        ax.set_ylim(bottom=0)



    if(len(ic50_samples[0]) > 5 and np.any(ic50_samples[0] < plot_lim)):
        ax2 = ax.twinx()
        ax2.hist(ic50_samples[0], bins=np.linspace(0, plot_lim, 50), color=['g'], histtype='step', normed=True,
            linewidth=1*lw_scale)
        ax2.set_yticks([])

    plt.xlim(-0.1, plot_lim + 0.1)

    ax.tick_params(axis='both', which='major', labelsize=labelsize)
