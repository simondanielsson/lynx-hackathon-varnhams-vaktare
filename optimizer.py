from collections import abc
import logging
from typing import List

import pandas as pd
import numpy as np
import scipy

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

def get_positions(position_model_name: str, datas: abc.Mapping[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Get portfolio positions using for each date.

    :param datas:
    :return: dataframe with positions for each asset for every date.
    """
    _log.info('Calculating positions...')
    position_model = POSITION_MODELS.get(position_model_name)

    if not position_model:
        raise ValueError(f'No model name with name {position_model}, not in {POSITION_MODELS.keys()}.')

    return position_model(datas, **kwargs)


def lynx_sign_model(datas: abc.Mapping[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    ret = datas['prices'].diff()

    vol_window = kwargs.get('vol_window', 50)
    trend_window = kwargs.get('trend_window', 100)

    pos = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    # loop over all dates
    for t in range(ret.shape[0]):
        # Volatility estimate; standard deviation on the last vol_window days, up to t-1
        vol = np.sqrt((ret ** 2).iloc[t - vol_window:t].mean())

        # Mean return between t-trend_window and t-1
        block_ret = ret.iloc[t - trend_window:t].sum()
        # Take a long position if the 50-days return is positive, otherwise take a short position (sign of the block return)
        unadj_pos = np.sign(block_ret)

        # Position at date t; risk adjust with volatility from previous date
        pos.iloc[t] = unadj_pos / vol

    return pos


def sharpe_optimizer(datas: abc.Mapping[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Optimize positions using predicted prices and historical asset price covariances."""
    # for result replication
    np.random.seed(1)

    n_assets = datas['prices'].shape[1]

    positions = []
    for args in zip(
        datas['predicted_prices'].iterrows(),
        datas['prices'].iterrows(),
        datas['covariance'],
    ):
        # random uniform initialization around 0
        x0 = np.random.rand(n_assets)

        position = scipy.optimize.minimize(
            _neg_predicted_sharpe_ratio_tomorrow,
            x0=x0,
            args=args,
            #bounds=[(-1, 1), (-1, 1)],
        )

        positions.append(position)


def _neg_predicted_sharpe_ratio_tomorrow(
    positions: np.array,  # variables to optimize
    predicted_prices_next: np.array,
    real_prices_yesterday: np.array,
    covariance_matrix: np.array,
) -> float:
    """Objective function to optimize for Sharpe ratio."""
    if not (len(positions) == len(predicted_prices_next) == len(real_prices_yesterday)):
        raise ValueError(f'lengths not the same: {len(positions)} {len(predicted_prices_next)} {len(real_prices_yesterday)}')

    returns = (predicted_prices_next - real_prices_yesterday).dot(positions)
    portfolio_std = np.sqrt(positions.T @ covariance_matrix @ positions)

    return - returns / portfolio_std


POSITION_MODELS = {
    'lynx_sign_model': lynx_sign_model,
    'sharpe_optimizer': sharpe_optimizer,
}