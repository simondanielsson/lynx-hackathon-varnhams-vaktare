from collections import abc
import logging

import pandas as pd
import numpy as np
from sklearn.covariance import ShrunkCovariance
from tqdm import tqdm

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def get_covariance(covariance_model_name: str, datas: abc.Mapping[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Get the covariance matrices for day t+1, determined at quote day t-1."""
    covariance_model = COVARIANCE_MODELS.get(covariance_model_name)

    if not covariance_model:
        raise ValueError(f'No covariance model with name {covariance_model_name},'
                         f' not in {COVARIANCE_MODELS.keys()}.')

    _log.info(f'Calculating return covariances using strategy `{covariance_model_name}`...')

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


def shrinkage(datas, **kwargs):
    cov_window_size = kwargs.get('cov_window_size')

    price_data_name = 'eval_prices' if kwargs.get('eval') else 'prices'
    returns = datas[price_data_name].set_index('dates').diff()

    covariance_matrices = []
    for t in tqdm(range(returns.shape[0]), total=returns.shape[0]):
        covariance_matrix = _get_shrunk_covariance_matrix(
            returns.iloc[t - cov_window_size:t]
        )
        date = returns.index[t]
        covariance_matrices.append((date, covariance_matrix))

    return covariance_matrices
            
    
def _get_shrunk_covariance_matrix(returns):
    # catch missing entries
    try:
        cov = ShrunkCovariance().fit(returns)
        return cov.covariance_

    except Exception as e:
        return np.array([None])


def no_op(datas, **kwargs):
    """Do no particular covariance estimation in this step."""
    return None


COVARIANCE_MODELS = {
    # name: cov_function
    'naive' : naive_covariance,
    'shrinkage': shrinkage,
    'no_op': no_op,
}