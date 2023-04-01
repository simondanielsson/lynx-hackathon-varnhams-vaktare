import json
import sys
from collections import abc
from datetime import datetime
from typing import Dict
import logging

import pandas as pd
import matplotlib.pyplot as plt

import evaluation
from covariance_model import get_covariance
from optimizer import get_positions
from predictions import PREDICTIONS_PATH
from price_model import get_prices

# TODO: fill in
NAMES_TO_FILE_NAMES = {
    "prices": "hackathon_prices_dev.csv",
}

_log = logging.getLogger(__name__)
logging.basicConfig()
_log.setLevel(logging.INFO)


def main(
    pricing_model_name: str,
    covariance_model_name: str,
    position_model_name: str,
    do_save: bool = False,
    **kwargs,
) -> int:
    _log.info(
        f'Launching trading strategy with the following config\n\tPricing model:{pricing_model_name}'
        f'\n\tCovariance mode: {covariance_model_name}\n\tPosition model: {position_model_name}\n\t'
        f"Hyperparameters: {'; '.join(f'{name}={value}'for name, value in kwargs.items())}"
    )
    datas: Dict[str, pd.DataFrame] = _load_data()

    datas['predicted_returns'] = get_prices(
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

    save_metadata = {
        'pricing_model_name': pricing_model_name,
        'covariance_model_name': covariance_model_name,
        'position_model_name': position_model_name,
        'hparams': kwargs,
    }
    _save_results(
        positions=positions,
        datas=datas,
        do_save=do_save,
        save_metadata=save_metadata,
    )

    return 0


def _load_data() -> abc.Mapping[str, pd.DataFrame]:
    """Load data from files."""
    _log.info(f'Loading data from {set(NAMES_TO_FILE_NAMES.keys())}')
    return {
        name: pd.read_csv(file_name)
        for name, file_name in NAMES_TO_FILE_NAMES.items()
    }


def _save_results(positions: pd.DataFrame, datas: abc.Mapping, do_save: bool, save_metadata: abc.Mapping) -> None:
    """Save positions to disk.

    NOTE: to load this csv file properly, for instance when used
    with `evaluation.plot_key_figures`, run

      `pd.read_csv(file_name, index_col='dates', parse_dates=['dates'])`.
    """
    if not do_save:
        _log.info(f'Skipped saving positions to disk, as `save_path` is {do_save}')
        return

    # prepare for prediction directory
    current_time = datetime.strftime(datetime.now(), format="%H%M")
    save_dir = (
        PREDICTIONS_PATH /
        f"{current_time}-{save_metadata['pricing_model_name']}-"
        f"{save_metadata['covariance_model_name']}-{save_metadata['position_model_name']}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    _log.info(f'Saving positions and metadata to {save_dir}')
    positions.to_csv(save_dir / 'positions.csv', index_label='dates')
    with open(save_dir / 'hyperparams.json', 'w') as fp:
        json.dump(save_metadata, fp)

    # save predicted prices
    datas['predicted_returns'].to_csv(save_dir / 'predicted_returns.csv', index_label='dates')

    # Performance panel
    prices = datas['prices'].set_index('dates')
    evaluation.plot_key_figures(positions, prices)
    plt.savefig(save_dir / 'performance_summary.png')


if __name__ == '__main__':
    """
    PRICE MODELS:
    'rolling_mean_returns'
    
    COVARIANCE MODELS:
    'naive'
    'shrinkage'
    'no_op'
    
    POSITION MODELS:
    'lynx_sign_model'
    'sharpe_optimizer'
    'package_sharpe_opt'
    """

    pricing_model_name = 'linear_return_predictor'
    covariance_model_name = 'no_op'
    position_model_name = 'no_op'
    do_save = True

    # add hyperparameters here! Make sure there are no name collisions
    kwargs = {
        'vol_window': 50,
        'trend_window': 100,
        'price_window_size': 10,
        'price_span': 50,
        'cov_window_size': 125,  # half a year
        # try these!
        'hist_window': 30,
        'future_window': 10,
        'vol_window': 100,
    }

    sys.exit(
        main(
            pricing_model_name=pricing_model_name,
            covariance_model_name=covariance_model_name,
            position_model_name=position_model_name,
            do_save=do_save,
            **kwargs,
        )
    )
