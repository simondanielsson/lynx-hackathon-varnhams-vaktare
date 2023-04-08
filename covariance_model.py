from collections import abc
import logging

import pandas as pd
import numpy as np
import scipy
from sklearn.covariance import ShrunkCovariance

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

def get_covariance(covariance_model_name: str, datas: abc.Mapping[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Get the covariance matrices for day t+1, determined at quote day t-1."""
    covariance_model = COVARIANCE_MODELS.get(covariance_model_name)

    if not covariance_model:
        raise ValueError(f'No covariance model with name {covariance_model_name},'
                         f' not in {COVARIANCE_MODELS.keys()}.')

    _log.info(f'Calculating asset covariances using strategy `{covariance_model_name}`...')

    return covariance_model(datas, **kwargs)


def naive_covariance(datas, **kwargs):
    cov_window_size = kwargs.get('cov_window_size')

    price_data_name = 'eval_prices' if kwargs.get('eval') else 'prices'
    returns = datas[price_data_name].set_index('dates').diff()

    # TODO: check if/where to do shift (1)
    covariances = returns.rolling(cov_window_size).cov()
    cov_matrices = []
    for date, cov_matrix_day_t in covariances.reset_index(level=[0, 1]).groupby('dates'):
        cov_matrix_day_t.drop(columns=cov_matrix_day_t.columns[0], axis=1, inplace=True)
        cov_matrices.append((date, cov_matrix_day_t))

    return cov_matrices


def ledoit_wolf(datas, **kwargs):
    cov_window_size = kwargs.get('cov_window_size')
    df = datas['prices'].set_index('dates')
    vols = []
    for t in range(df.shape[0]-1):
        vol = _cov_shrunk(df.iloc[t-cov_window_size: t])
        vols.append(vol)
    return vols
            
    
def _cov_shrunk(x):
    # catch missing entries
    try:
        cov = ShrunkCovariance().fit(x)
        return cov.covariance_
    except:
        return None


def no_op(datas, **kwargs):
    """Do no particular covariance estimation in this step."""
    return None


COVARIANCE_MODELS = {
    # name: cov_function
    'naive' : naive_covariance,
    'shrinkage': ledoit_wolf,
    'no_op': no_op,
}