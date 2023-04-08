from collections import abc
import logging

import pandas as pd
import numpy as np
import pypfopt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

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
    price_span = kwargs.get('price_span', 50)

    predicted_returns = []
    for index in tqdm(range(30, prices.shape[0] - 1), total=prices.shape[0]):
        prices_until_incl_quote = prices.iloc[:index+1, :]

        #ret = prices_until_incl_quote.diff()
        #vol = np.sqrt((ret ** 2).rolling(window=30, min_periods=10).mean())
        #norm_ret = ret / vol

        # default span=500
        predicted_returns_at_trade = pypfopt.expected_returns.ema_historical_return(
            prices_until_incl_quote,
            #norm_ret.dropna(),
            span=price_span,
            #returns_data=True,
        ) * 252

        predicted_returns_at_trade_df = pd.DataFrame(
            predicted_returns_at_trade,
        ).T
        predicted_returns.append(predicted_returns_at_trade_df)

    return pd.concat(predicted_returns)


def linear_model_window_return_predictor(datas, **kwargs):
    models = train_models(
        datas['prices'].set_index('dates'),
        kwargs.get('hist_window'),
        kwargs.get('future_window')
    )

    price_data_name = 'eval_prices' if kwargs.get('eval') else 'prices'
    returns = get_returns_from_linear_model(
        datas[price_data_name].set_index('dates'),
        models,
        kwargs.get('hist_window'),
        kwargs.get('vol_window'),
    )

    return returns


def lynx_predictor_with_linear_returns(datas, **kwargs):
    models = train_models(
        datas['prices'].set_index('dates'),
        kwargs.get('hist_window'),
        kwargs.get('future_window')
    )

    price_data_name = 'eval_prices' if kwargs.get('eval') else 'prices'
    positions = get_positions_from_linear_model_lynx(
        datas[price_data_name].set_index('dates'),
        models,
        kwargs.get('hist_window'),
        kwargs.get('vol_window'),
    )

    return positions


def train_models(prices, hist_window=30, future_window=10):
    """Train a linear model using historical returns, predicting accumulated future_window days return"""
    ret = prices.ffill().diff().dropna()

    models = {column: LinearRegression() for column in ret.columns}
    Xs = {column: [] for column in ret.columns}
    ys = {column: [] for column in ret.columns}

    _log.info('Aquiring training data...')
    for column, model in models.items():
        for index in tqdm(range(hist_window, ret.shape[0]), total=ret.shape[0]):
            X_cur = ret[column].dropna().iloc[index - hist_window:index].to_numpy()
            y_cur = ret[column].dropna().iloc[index:index + future_window].cumsum().to_numpy()[-1]

            Xs[column].append(X_cur)
            ys[column].append(y_cur)

    _log.info('Fitting models...')
    for column, model in tqdm(models.items()):
        model.fit(Xs[column], ys[column])

    return models


def get_returns_from_linear_model(prices, models, trend_window=30, vol_window=100):
    ret = prices.ffill().diff().dropna()

    #pos = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)

    _log.info('Inferring returns...')
    predicted_future_window_returns = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    for t in tqdm(range(trend_window, ret.shape[0] - 1), total=ret.shape[0]):
        # predict coming future_days accumulated days return
        predicted_future_window_return_trade_day = pd.DataFrame(
            {column: model.predict([ret[column].iloc[t - trend_window:t]]) for column, model in models.items()}
        )

        # predicted return for future_window day holding, from [t, t+future_window), using info from t-1 and earlier
        # return at trade day t, i.e. price change between t to t+future_window.
        predicted_future_window_returns.iloc[t+1] = predicted_future_window_return_trade_day

    return predicted_future_window_returns


def get_positions_from_linear_model_lynx(prices, models, trend_window=30, vol_window=100):
    ret = prices.ffill().diff().dropna()

    pos = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    for t in tqdm(range(trend_window, ret.shape[0] - 1), total=ret.shape[0]):
        # Volatility estimate; standard deviation on the last vol_window days, up to t-1
        vol = np.sqrt((ret ** 2).iloc[t - vol_window:t].mean())

        # predict coming future_days accumulated days return
        predicted_future_window_return_trade_day = pd.DataFrame(
            {column: model.predict([ret[column].iloc[t - trend_window:t]]) for column, model in models.items()}
        )

        # Take a long position if the historical 30-days predicted 10-day return is positive, otherwise take a short position
        # scale by return magnitude
        # Position at date t; risk adjust with volatility from previous date
        pos.iloc[t + 1] = predicted_future_window_return_trade_day / vol

    return pos


PRICING_MODELS = {
    'same_as_yesterday': same_as_yesterday_model,
    'rolling_mean_returns': rolling_mean_returns,
    'ema_returns': ema_returns,
    'linear_return_predictor': linear_model_window_return_predictor,
    'linear_return_independent_assets_lynx': lynx_predictor_with_linear_returns ,
}