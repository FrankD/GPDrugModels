# Script for fitting responder GP model to specific drugs/cell lines for exploration.

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = (12, 7)
from matplotlib import pyplot as plt
plt.style.use('seaborn-talk')
import gpdm
import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf


plotting_dir = 'figures/'
label = 'fit'

# Specify dose-response curves that need to be fitted 
to_fit = pd.read_csv('data/curves_to_fit.csv')

# Dose-response data
data = gpdm.test_data.process_data('data/v17a_public_raw_data_with_drug_names.csv')

# Dataframe storing results
summary_measures = pd.DataFrame(columns=['Drug', 'Drug_Name', 'Cell_Line',
                                         'AUC', 'AUC_Std', 
                                         'IC50', 'IC50_Std', 'p_IC50',
                                         'LL'])


num_curves = to_fit.shape[0]

# Loop over dose-response curves
for curve_count, fit_curve in to_fit.iterrows():  
    
    # Locate raw data to fit
    drug = fit_curve.DRUG_ID
    row = data.loc[(data.DRUG_ID == drug) & (data.COSMIC_ID == fit_curve.COSMIC_ID)]


    if row.shape[0]==1: # Data found
        row = row.iloc[0]

        message = str(curve_count+1) + "/" + str(num_curves) + "\n" 
        print(message)

        drug_name = row.DRUG_NAME
        cl_name = row.CELL_LINE_NAME

        # Put data in right format.
        X, Y = row.X.reshape(-1), row.Y.reshape(-1)
        X = (X + 1.0) / (np.max(X) + 1.0) # Rescale

        m = gpdm.models.responder_model(X, Y)
        
        # Set lengthscale for GP
        m.kern.kernels[0].lengthscales = .3
        
        # Optimize
        opt = gpflow.training.ScipyOptimizer()
        m.compile()        
        opt.minimize(m)
        
        # Calculate summary measures (AUC and IC50)
        auc_mean, auc_std = gpdm.summary_measures.get_AUC_stats(m, 1000)

        p_ic50, ic50_samples = gpdm.summary_measures.get_ic50_samples(m, 5000)
        ic50_mean = np.mean(ic50_samples)
        ic50_std = np.std(ic50_samples) 
        
        # Plot curve fit
        gpdm.plotting.plot(m, ic50_samples=ic50_samples)
        plt.savefig(plotting_dir + '%s_%s_%s.png' % (drug, row.COSMIC_ID, label))
        plt.close()

        ll = m.compute_log_likelihood()

    else: # Data not found
        drug_name=fit_cl.drug_name
        cl_name=fit_cl.CELL_LINE_NAME
        auc_mean=-1
        auc_std=-1
        ic50_mean=-1
        ic50_std=-1
        p_ic50 = [-1]
        ll = -1
        
    summary_measures = summary_measures.append([{'Drug':drug, 'Drug_Name':drug_name,
                                                     'Cell_Line':cl_name,
                                                    'AUC':auc_mean,'AUC_Std':auc_std,                                                    
                                                    'IC50':ic50_mean, 'IC50_Std':ic50_std,
                                                    'p_IC50':p_ic50[0], 
                                                    'LL':ll}])
        
summary_measures.to_csv('curve_fits_summary_measures.csv')




