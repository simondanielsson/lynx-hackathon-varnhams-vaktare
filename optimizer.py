import copy
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
    # remove first element as no return info exists for that date
    covariance_matrices = datas['covariance'][1:]
    predicted_returns = datas['predicted_returns']

    if len(predicted_returns) != len(covariance_matrices):
        raise ValueError(
            f"Predicted returns and covariance dataframe is not of equal length: "
            f"{len(predicted_returns)} != {len(covariance_matrices)}"
        )

    # for result replication
    np.random.seed(1)

    # exclude date column
    n_assets = datas['prices'].shape[1] - 1
    assets = datas['prices'].columns[1:]

    # display how many dates first were skipped
    n_date_skips = 0
    check_nans = True

    # track failed convergences
    failed_count = 0

    # skip updating positions every update_freq'th day
    skip_update_freq = kwargs.get('skip_update_freq')
    counter = 1
    nbr_skips = 0

    positions = []
    for (
        (return_date, predicted_returns_trade_day),
        (cov_date, covariance_matrix),
    ) in tqdm(
        zip(
            predicted_returns.iterrows(),
            covariance_matrices,
        ),
        total=len(datas['predicted_returns'])
    ):
        if not return_date == cov_date:
            raise ValueError(f"Predicted return date not aligned with cov date: {return_date} != {cov_date}.")

        if counter % skip_update_freq == 0:
            # stay at same position
            last_pos = copy.deepcopy(positions[-1])
            last_pos.index = [cov_date]
            positions.append(last_pos)
            nbr_skips += 1
            counter += 1
            continue

        if isinstance(covariance_matrix, pd.DataFrame):
            covariance_matrix_np = covariance_matrix.set_index('level_1').to_numpy()
        elif isinstance(covariance_matrix, np.ndarray):
            covariance_matrix_np = covariance_matrix
        else:
            raise ValueError(f'Could not resolve type of covariance matrix: type is {type(covariance_matrix)}')

        # first iteration when no previous positions exist, we don't compute any slippage
        positions_yesterday = positions[-1].to_numpy().T if positions else None

        optimization_args = (
            predicted_returns_trade_day.to_numpy(),
            positions_yesterday,
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


        # TODO: try initializing from yesterdays position!!
        # random uniform position initialization in [-1, 1]
        #x0 = np.random.rand(n_assets) * 2 - 1
        x0 = predicted_returns_trade_day.to_numpy()

        position = scipy.optimize.minimize(
            _neg_predicted_sharpe_ratio_tomorrow,
            x0=x0,
            args=optimization_args,
        )

        if not position.success:
            failed_count += 1

        position_np = np.array(position.x).reshape(1, -1)

        position_df = pd.DataFrame(position_np, columns=assets, index=[return_date])

        positions.append(position_df)
        counter += 1

    _track_progress(failed_count=failed_count, total_count=len(datas['prices']), nbr_skips=nbr_skips)
    # TODO: check if we should shift this by 1
    return pd.concat(
        positions,
    )
    #).shift(1)


def _any_contains_nan(args: Tuple[np.array]) -> bool:
    return any(np.isnan(arg).any() for arg in args if isinstance(arg, np.ndarray))


def _track_progress(failed_count: int, total_count: int, nbr_skips: int) -> None:
    if failed_count:
        _log.info(f"Position optimizer did not converge for {failed_count} / {total_count} dates; {nbr_skips=}")
    else:
        _log.info(f"Optimizer converged for all dates! {nbr_skips=}")


def _neg_predicted_sharpe_ratio_tomorrow(
    positions: np.array,  # variables to optimize
    predicted_returns_quote: np.array,
    positions_yesterday: np.array,
    covariance_matrix: np.array,
) -> float:
    """Objective function to optimize for Sharpe ratio."""
    # TODO: try penalising squared change in position
    #slippage = 0.0002 * np.abs(positions - positions_yesterday) if positions_yesterday is not None else 0.0
    slippage = 10 * 0.0002 * np.square(positions - positions_yesterday) if positions_yesterday is not None else 0.0

    returns = predicted_returns_quote.dot(positions) - np.sum(slippage)
    portfolio_std = np.sqrt(positions.T @ covariance_matrix @ positions)

    return - returns / portfolio_std


def package_sharpe_opt(datas: abc.Mapping[str, pd.DataFrame], **kwargs):
    """Optimize daily positions for Sharpe ratio.

    Alias `package_sharpe_opt`.

    :param datas: dataframes with technical indicators over time.
    :param kwargs: optional hyperparameters.
    :return: asset positions over time.
    """
    prices = datas['prices'].set_index('dates')

    positions = []
    for index in range(kwargs.get('cov_window_size'), prices.shape[0]):
        covariance_matrix = datas['covariance'][index].set_index('level_1').to_numpy()
        predicted_returns = datas['predicted_returns'].iloc[index, :]

        ef = EfficientFrontier(predicted_returns, cov_matrix=covariance_matrix)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        position = pd.DataFrame(cleaned_weights, index=[prices.index[index]])

        positions.append(position)

    return pd.concat(
        positions
    ).shift(1)  # align with trade date


def get_positions_from_linear_model(datas, **kwargs):
    """Get positions already computed by the linear model in price_model.py."""
    return datas['predicted_returns']


POSITION_MODELS = {
    'lynx_sign_model': lynx_sign_model,
    'sharpe_optimizer': sharpe_optimizer,
    'package_sharpe_opt': package_sharpe_opt,
    'no_op': get_positions_from_linear_model
}