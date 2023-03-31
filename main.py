import sys
from collections import abc
from typing import Dict, Optional
import logging

import pandas as pd
import evaluation

from covariance_model import get_covariance
from optimizer import get_positions
from price_model import get_prices

# TODO: fill in
NAMES_TO_FILE_NAMES = {
    "prices": "example_prices.csv",  # TODO: change
}

_log = logging.getLogger(__name__)
logging.basicConfig()
_log.setLevel(logging.INFO)


def main(
    pricing_model_name: str,
    covariance_model_name: str,
    position_model_name: str,
    save_path: Optional[str] = None,
    **kwargs,
) -> int:
    _log.info(
        f'Launching trading strategy with the following config\n\tPricing model:{pricing_model_name}'
        f'\n\tCovariance mode: {covariance_model_name}\n\tPosition model: {position_model_name}\n\t'
        f"Hyperparameters: {'; '.join(f'{name}={value}'for name, value in kwargs.items())}"
    )
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

    _save_results(positions=positions, save_path=save_path)

    return 0


def _load_data() -> abc.Mapping[str, pd.DataFrame]:
    """Load data from files."""
    _log.info(f'Loading data from {set(NAMES_TO_FILE_NAMES.keys())}')
    return {
        name: pd.read_csv(file_name)
        for name, file_name in NAMES_TO_FILE_NAMES.items()
    }


def _save_results(positions: pd.DataFrame, save_path: Optional[str]) -> None:
    """Save positions to disk."""
    if not save_path:
        _log.info(f'Skipped saving positions to disk, as `save_path` is {save_path}')
        return

    _log.info(f'Saving positions to {save_path}')
    positions.to_csv(save_path, index_label='dates')


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

    pricing_model_name = 'rolling_mean_price'
    covariance_model_name = 'naive'
    position_model_name = 'sharpe_optimizer'
    save_path = 'predictions/first_pred.csv'

    # add hyperparameters here! Make sure there are no name collisions
    kwargs = {
        'vol_window': 50,
        'trend_window': 100,
        'price_window_size': 10,
        'cov_window_size': 20,
    }

    sys.exit(
        main(
            pricing_model_name=pricing_model_name,
            covariance_model_name=covariance_model_name,
            position_model_name=position_model_name,
            save_path=save_path,
            **kwargs,
        )
    )
