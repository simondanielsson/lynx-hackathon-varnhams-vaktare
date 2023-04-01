from collections import abc
import logging

import pandas as pd
import numpy as np
import scipy

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
    df = datas['prices'].set_index('dates')
    # TODO: check where to do shift (1)
    # TODO: remove diff if we are working with returns
    # df = df.diff()
    covariances = df.rolling(cov_window_size).cov()
    matrices = []
    for date, new_df in covariances.reset_index(level=[0,1]).groupby('dates'):
        new_df.drop(columns=new_df.columns[0], axis=1, inplace=True)
        matrices.append(new_df)

    return matrices


COVARIANCE_MODELS = {
    # name: cov_function
    'naive' : naive_covariance
}