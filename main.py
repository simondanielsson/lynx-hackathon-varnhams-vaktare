import sys
from collections import abc
from typing import Dict, Optional
import logging

import pandas as pd
import numpy as np
import evaluation
import scipy

from covariance_model import get_covariance
from optimizer import get_positions
from price_model import get_prices

# TODO: fill in
NAMES_TO_FILE_NAMES = {
    "prices": "example_prices.csv",  # TODO: change
}

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def main(
    pricing_model_name: str,
    covariance_model_name: str,
    position_model_name: str,
    save_path: Optional[str] = None,
    **kwargs,
) -> int:
    datas: Dict[str, pd.DataFrame] = _load_data()

    datas['predicted_prices'] = get_prices(
        pricing_model_name=pricing_model_name,
        datas=datas,
        **kwargs,
    )

    datas['covariance'] = get_covariance(
        covariance_model_name=covariance_model_name,
        datas=datas,
        **kwargs,
    )

    positions = get_positions(
        position_model_name=position_model_name,
        datas=datas,
        **kwargs,
    )

    _save(positions)

    return 0


def _load_data() -> abc.Mapping[str, pd.DataFrame]:
    """Load data from files."""
    _log.info(f'Loading data from {NAMES_TO_FILE_NAMES.keys()}')
    return {
        name: pd.read_csv(file_name)
        for name, file_name in NAMES_TO_FILE_NAMES.items()
    }


def _save(positions: pd.DataFrame, save_path: Optional[str]) -> None:

    if save_path:
        _log.info(f'Saving positions to {save_path}')
        positions.to_csv(save_path)


if __name__ == '__main__':
    """
    PRICE MODELS:
    'rolling_mean_price'
    
    COVARIANCE MODELS:
    'naive'
    
    POSITION MODELS:
    'lynx_sign_model'
    'sharpe_optimizer'
    """

    pricing_model_name = ""
    covariance_model_name = ""
    position_model_name = ""
    save_path = None

    # add hyperparameters here! Make sure there are no name collisions
    kwargs = {
        'vol_window': 50,
        'trend_window': 100,
        'price_window_size': 10,
        'cov_window_size' : 20
    }

    sys.exit(
        main(
            pricing_model_name,
            covariance_model_name,
            position_model_name,
            **kwargs,
        )
    )
