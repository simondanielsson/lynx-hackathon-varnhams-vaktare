from collections import abc
import logging

import pandas as pd
import pypfopt
from tqdm import tqdm

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def get_prices(pricing_model_name: str, datas: abc.Mapping[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Return predicted prices for all dates, using data available in datas.

    :param datas: dict of dataframes with prices and other data.
    :returns: dataframe with predicted prices for all assets and timestamps.
    """
    pricing_model = PRICING_MODELS.get(pricing_model_name)

    if not pricing_model:
        raise ValueError(f'No model name with name {pricing_model_name}, not in {PRICING_MODELS.keys()}.')

    _log.info(f'Predicting prices using model type `{pricing_model_name}`')

    return pricing_model(datas, **kwargs)


def same_as_yesterday_model(datas: abc.Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Predict prices as same as yesterday."""
    pass

def arma_price(datas, **kwargs):
    p = kwargs.get('p', 1)
    q = kwargs.get('q', 1)
    ws = kwargs.get('window_size', 10)


def rolling_mean_returns(datas, **kwargs):
    prices = datas['prices'].set_index('dates')

    nbr_days = kwargs.get('price_window_size', 10)

    # assume quote day is today, i.e. we know the price today and set our order the coming night
    predicted_price_1_days = prices.diff().rolling(nbr_days).mean().shift(1)  # [.., quote - 1]
    predicted_price_2_days = prices.diff().rolling(nbr_days).mean()           # [.., quote]

    predicted_return_on_quote_day = predicted_price_2_days - predicted_price_1_days

    return predicted_return_on_quote_day


def ema_returns(datas, **kwargs):
    """
    Calculate exponentially-weighted mean of daily returns.

    :param datas:
    :param kwargs:
    :return:
    """
    prices = datas['prices'].set_index('dates')

    predicted_returns = []
    for index in tqdm(range(20, prices.shape[0] - 1), total=prices.shape[0]):
        prices_until_incl_quote = prices.iloc[:index+1, :]

        # default span=500
        predicted_returns_at_trade = pypfopt.expected_returns.ema_historical_return(
            prices_until_incl_quote
        )

        predicted_returns_at_trade_df = pd.DataFrame(
            predicted_returns_at_trade,
        ).T
        predicted_returns.append(predicted_returns_at_trade_df)

    return pd.concat(predicted_returns)


# TODO: insert your function here with a name
PRICING_MODELS = {
    'same_as_yesterday': same_as_yesterday_model,
    'rolling_mean_returns': rolling_mean_returns,
    'ema_returns': ema_returns
}