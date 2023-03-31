from collections import abc
import logging

import pandas as pd
import numpy as np
import scipy

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

def get_covariance(covariance_model_name: str, datas: abc.Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Get the covariance for a """
    covariance_model = COVARIANCE_MODELS.get(covariance_model_name)

    if not covariance_model:
        raise ValueError(f'No covariance model with name {covariance_model_name},'
                         f' not in {COVARIANCE_MODELS.keys()}.')

    _log.info(f'Calculating asset covariances using {covariance_model_name}...')

    return covariance_model(datas)



COVARIANCE_MODELS = {
    # name: cov_function
    'naive' : naive_covariance
}

def naive_covariance(datas):
  return datas['prices'].cov() # Hard coded to prices for now :)