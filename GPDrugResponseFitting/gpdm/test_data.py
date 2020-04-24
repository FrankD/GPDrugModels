import numpy as np
import pandas as pd
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
from . import models


def process_data(filename=None):
    if filename is None:
        filename = os.path.join(dir_path, 'data/sanger_public_data_test.csv')
    data = pd.read_csv(filename)
    
    blanks = data[[n for n in data.columns if 'blank' in n]]
    control = data[[n for n in data.columns if 'control' in n]]
    
    # Dealing with string entries saying 'qc_fail' -> turn them into nan
    blanks = blanks.apply(pd.to_numeric, errors='coerce') 
    control = control.apply(pd.to_numeric, errors='coerce')

    data['blankmean'] = np.nanmean(blanks, 1)
    data['controlmean'] = np.nanmean(control, 1)

    data['X'] = [np.arange(9) if fd == 2 else np.arange(5) for fd in data.FOLD_DILUTION]
    Y = data[['raw' + str(i) for i in [9, 8, 7, 6, 5, 4, 3, 2, '_max']]]
    data['Y'] = [(y-bm)/(cm-bm) if fd == 2 else (y[4:] - bm)/(cm-bm)
                 for y, fd, bm, cm in zip(Y.values, data.FOLD_DILUTION, data.blankmean, data.controlmean)]

    return data[['CELL_LINE_NAME', 'COSMIC_ID', 'DRUG_ID', 'DRUG_NAME', 'X', 'Y', 'MAX_CONC']]


def process_data_replicates(filename=None):
    if filename is None:
        filename = os.path.join(dir_path, 'data/monotherapy_drug_data_replicates.csv')
    data = pd.read_csv(filename)
    
    data['X'] = [np.arange(7) for bc in data.BARCODE]
    Y = data[['raw_' + str(i) for i in [6, 5, 4, 3, 2, 1, 0]]]
    data['Y'] = [(y-bm)/(cm-bm) 
                 for y, bm, cm in zip(Y.values, data.CNTRL_blank, data.CNTRL_neg)]

    return data[['BARCODE', 'CELL_LINE_NAME', 'COSMIC_ID', 'DRUG_ID', 'DRUG_NAME', 'X', 'Y', 'MAX_CONC']]

def process_data_CTD2(filename=None):
    if filename is None:
        filename = os.path.join(dir_path, 'data/v20.data.per_cpd_well.dabrafenib.tsv')
    data = pd.read_table(filename)
    
    out_data = pd.DataFrame()

    for experiment in np.unique(data.experiment_id):
        data_temp = data[data.experiment_id==experiment]
        for bc in np.unique(data_temp.assay_plate_barcode):
           data_subset = data_temp[data.assay_plate_barcode==bc]
           temp_data = pd.DataFrame()
           temp_data['X'] = [np.round(np.log2(data_subset.cpd_conc_umol)+9)]
           temp_data['Y'] = [np.power(2, data_subset.bsub_value_log2)]
           temp_data['BARCODE'] = bc
           temp_data['CELL_LINE_NAME'] = data_subset.ccl_name.values[0]
           temp_data['DRUG_ID'] = data_subset.master_cpd_id.values[0]
           temp_data['DRUG_NAME'] = data_subset.master_cpd_id.values[0]
           temp_data['MAX_CONC'] = 66
           out_data = out_data.append(temp_data)
    
    return out_data


def run_in_serial():
    data = process_data()
    ll_responder = []
    ll_nonresponder = []
    for i, row in data.iterrows():
        X, Y = row.X.reshape(-1, 1), row.Y.reshape(-1, 1)
        m = models.responder_model(X, Y)
        m.optimize()
        m_null = models.nonresponder_model(X, Y)
        m_null.optimize()
        ll_responder.append(m.compute_log_likelihood())
        ll_nonresponder.append(m_null.compute_log_likelihood())
    
    data['gamma'] = np.array(ll_responder).squeeze()
    data['zeta'] = np.array(ll_nonresponder).squeeze()

    return data
