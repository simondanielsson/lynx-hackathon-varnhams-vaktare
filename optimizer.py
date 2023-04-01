from collections import abc
import logging
from typing import List, Tuple

import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from pypfopt.efficient_frontier import EfficientFrontier

from utils import timing

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

def get_positions(position_model_name: str, datas: abc.Mapping[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Get portfolio positions using for each date.

    :param datas:
    :return: dataframe with positions for each asset for every date.
    """
    position_model = POSITION_MODELS.get(position_model_name)

    if not position_model:
        raise ValueError(f'No model name with name {position_model}, not in {POSITION_MODELS.keys()}.')

    _log.info(f"Calculating positions using strategy `{position_model_name}`")

    return position_model(datas, **kwargs)


@timing
def lynx_sign_model(datas: abc.Mapping[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    ret = datas['prices'].set_index('dates').diff()

    vol_window = kwargs.get('vol_window', 50)
    trend_window = kwargs.get('trend_window', 100)

    pos = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    # loop over all dates
    for t in tqdm(range(ret.shape[0]), total=ret.shape[0]):
        # Volatility estimate; standard deviation on the last vol_window days, up to t-1
        vol = np.sqrt((ret ** 2).iloc[t - vol_window:t].mean())

        # Mean return between t-trend_window and t-1
        block_ret = ret.iloc[t - trend_window:t].sum()
        # Take a long position if the 50-days return is positive, otherwise take a short position (sign of the block return)
        unadj_pos = np.sign(block_ret)

        # Position at date t; risk adjust with volatility from previous date
        pos.iloc[t] = unadj_pos / vol

    return pos


@timing
def sharpe_optimizer(datas: abc.Mapping[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Optimize positions using predicted prices and historical asset price covariances."""
    # for result replication
    np.random.seed(1)

    # exclude date column
    n_assets = datas['prices'].shape[1] - 1

    # display how many dates first were skipped
    n_date_skips = 0
    check_nans = True

    # track failed convergences
    failed_count = 0

    positions = []
    for (
        predicted_returns_quote,
        covariance_matrix,
    ) in tqdm(zip(
        datas['predicted_returns'].iterrows(),
        datas['covariance'],
    ), total=len(datas['prices'])):
        #real_prices_yesterday_np = real_prices_yesterday[1].to_numpy()

        # make sure predicted and real dates are aligned
        # _verify_date(predicted_returns_next, real_prices_yesterday_np)

        if isinstance(covariance_matrix, pd.DataFrame):
            covariance_matrix_np = covariance_matrix.set_index('level_1').to_numpy()
        elif isinstance(covariance_matrix, np.ndarray):
            covariance_matrix_np = covariance_matrix
        else:
            raise ValueError(f'Could not resolve type of covariance matrix: type is {type(covariance_matrix)}')

        optimization_args = (
            predicted_returns_quote[1].to_numpy(),
            #real_prices_yesterday_np[1:].astype(dtype=np.float64),  # remove date column
            covariance_matrix_np,
        )

        # skip if any of the inputs contains nan's; if so, the window
        # sizes prevents us for predicting one these dates
        if check_nans:
            if _any_contains_nan(optimization_args):
                n_date_skips += 1
                continue

        if n_date_skips and check_nans:
            check_nans = False
            _log.info(f'Skipped the {n_date_skips} first dates in optimizer.'
                      f' Make sure this aligns with the maximal window size')

        # random uniform position initialization in [-1, 1]
        x0 = np.random.rand(n_assets) * 2 - 1

        position = scipy.optimize.minimize(
            _neg_predicted_sharpe_ratio_tomorrow,
            x0=x0,
            args=optimization_args,
            #bounds=[(-1, 1), (-1, 1)],
        )

        if not position.success:
            failed_count += 1
            #

        position_np = np.array(position.x).reshape(1, -1)

        position_df = pd.DataFrame(position_np, columns=datas['prices'].columns[1:], index=[predicted_returns_quote[0]])

        positions.append(position_df)

    _track_progress(failed_count=failed_count, total_count=len(datas['prices']))
    # indexed by date; shifted up to trade date (i.e. from quote day to trade day, pos_t)
    return pd.concat(
        positions,
    ).shift(1)


def _verify_date(predicted_prices_next: List, real_prices_yesterday_np: np.array) -> None:
    predicted_price_date = predicted_prices_next[0]
    real_prices_date = real_prices_yesterday_np[0]
    assert predicted_price_date == real_prices_date


def _any_contains_nan(args: Tuple[np.array]) -> bool:
    return any(np.isnan(arg).any() for arg in args)


def _track_progress(failed_count: int, total_count: int) -> None:
    if failed_count:
        _log.info(f"Position optimizer did not converge for {failed_count} / {total_count} dates")
    else:
        _log.info(f"Optimizer converged for all dates!")


def _neg_predicted_sharpe_ratio_tomorrow(
    positions: np.array,  # variables to optimize
    predicted_returns_quote: np.array,
    covariance_matrix: np.array,
) -> float:
    """Objective function to optimize for Sharpe ratio."""
    if len(positions) != len(predicted_returns_quote):
        raise ValueError(f'lengths not the same: {len(positions)} != {len(predicted_returns_quote)}')

    slippage = 0.0002 *
    returns = predicted_returns_quote.dot(positions) - slippage
    portfolio_std = np.sqrt(positions.T @ covariance_matrix @ positions)

    return - returns / portfolio_std


def package_sharpe_opt(datas: abc.Mapping[str, pd.DataFrame], **kwargs):
    """Optimize daily positions for Sharpe ratio.

    :param datas: dataframes with technical indicators over time.
    :param kwargs: optional hyperparameters.
    :return: asset positions over time.
    """
    prices = datas['price'].set_index('dates')

    positions = []
    for index in range(prices.shape[0]):
        covariance_matrix = datas['covariance'][index]
        predicted_returns = datas['predicted_returns'].iloc[index, :]

        ef = EfficientFrontier(predicted_returns, cov_matrix=covariance_matrix)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        position = pd.DataFrame(cleaned_weights, columns=[prices.columns], index=[prices.index[index]])

        positions.append(cleaned_weights)


    return pd.concat(
        positions
    )


POSITION_MODELS = {
    'lynx_sign_model': lynx_sign_model,
    'sharpe_optimizer': sharpe_optimizer,
    'package_sharpe_opt': package_sharpe_opt,
}