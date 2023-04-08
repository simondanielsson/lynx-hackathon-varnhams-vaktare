import json
import sys
from collections import abc
from datetime import datetime
from typing import Dict
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import evaluation
from covariance_model import get_covariance
from optimizer import get_positions
from predictions import PREDICTIONS_PATH
from predictions_after import PREDICTIONS_AFTER_PATH
from price_model import get_prices

# TODO: fill in
NAMES_TO_FILE_NAMES = {
    "prices": Path('data') / 'hackathon_prices_dev.csv',
    "eval_prices": Path('data') / 'hackathon_prices_all.csv',
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
        **kwargs,
    )

    return 0


def _load_data() -> abc.Mapping[str, pd.DataFrame]:
    """Load data from files."""
    _log.info(f'Loading data from {set(NAMES_TO_FILE_NAMES.keys())}')
    return {
        name: pd.read_csv(file_name)
        for name, file_name in NAMES_TO_FILE_NAMES.items()
    }


def _save_results(positions: pd.DataFrame, datas: abc.Mapping, do_save: bool, save_metadata: abc.Mapping, **kwargs) -> None:
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
        PREDICTIONS_AFTER_PATH /
        f"{current_time}{'-eval' if kwargs.get('eval') else ''}-{save_metadata['pricing_model_name']}-"
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
    price_data_name = 'eval_prices' if kwargs.get('eval') else 'prices'
    prices = datas[price_data_name].set_index('dates')
    evaluation.plot_key_figures(positions, prices)
    plt.savefig(save_dir / 'performance_summary.png')


if __name__ == '__main__':
    """
    PRICE MODELS:
    'rolling_mean_returns'
    'linear_return_predictor'
    'linear_return_independent_assets_lynx'
    'lynx_baseline'
    
    COVARIANCE MODELS:
    'naive'
    'shrinkage'
    'no_op'
    
    POSITION MODELS:
    'lynx_sign_model'
    'sharpe_optimizer'
    'package_sharpe_opt'
    'no_op'
    """

    pricing_model_name = 'linear_return_predictor'
    covariance_model_name = 'shrinkage'
    position_model_name = 'sharpe_optimizer'
    do_save = True

    # add hyperparameters here! Make sure there are no name collisions
    kwargs = {
        'trend_window': 100,
        'price_window_size': 10,
        'price_span': 50,
        'cov_window_size': 125,  # half a year
        # try these!
        'hist_window': 150,
        'future_window': 6,
        'vol_window': 100,
        'skip_update_freq': 3,  # skip update position frequency
        'eval': True,
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
