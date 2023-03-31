from collections import abc
import logging

import pandas as pd

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

    _log.info(f'Predicting prices using model type {pricing_model_name}')

    return pricing_model(datas, **kwargs)


def same_as_yesterday_model(datas: abc.Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Predict prices as same as yesterday."""
    pass

def arma_price(datas, **kwargs):
    p = kwargs.get('p', 1)
    q = kwargs.get('q', 1)
    ws = kwargs.get('window_size', 10)


def rolling_mean_price(datas, **kwargs):
    nbr_days = kwargs.get('price_window_size', 10)
    prediction = datas['prices'].rolling(nbr_days).mean().shift(1)
    return prediction


# TODO: insert your function here with a name
PRICING_MODELS = {
    'same_as_yesterday': same_as_yesterday_model,
}