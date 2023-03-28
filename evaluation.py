import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_KEY_FIGURES = [
    "av_holding",
    "kurtosis",
    "position_bias",
    "sharpe",
    "sharpe_1d_lag",
    "skewness",
]

ALL_KEY_FIGURES = [
    "av_holding",
    "av_holding_per_instrument",
    "kurtosis",
    "position_bias",
    "position_bias_per_instrument",
    "returns",
    "sharpe",
    "sharpe_1d_lag",
    "sharpe_ex_slippage",
    "sharpe_per_instrument",
    "skewness",
    "slippages",
    "rolling_position_bias",
    "rolling_sharpe",
    "rolling_std",
]


def calc_key_figures(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    costs=-0.02,
    rolling_window=260,
    key_figures=None,
):
    """
    Calculates key figures for a set of positions for a trading strategy. Each key figure is
    calculated over the full input period.

    Returns a dictionary with one or more of the following key figures:
        av_holding:
            Measures how long a position is held on average, once taken. Faster models (with low
            average holding periods) will incur more simulated slippage and may be harder to trade
            at size.

        av_holding_per_instrument:
            av_holding for each individual instrument.

        kurtosis
            Daily kurtosis of the estimated total returns of the strategy (including slippage).
            Values returned are kurtosis as opposed to excess kurtosis, meaning that normally
            distributed returns will have a kurtosis of 3.

        position_bias
            Indicates whether the strategy tended to hold long positions or short positions.
            The key figure returns values in the range [-1, 1] where -1 is short-only positions and
            vice versa.

        position_bias_per_instrument
            position_bias for each individual instrument.

        returns:
            Backtested returns per instrument and day.

        sharpe:
            Yearly risk adjusted return, including slippage. For simplicity no adjustment is made
            for the risk-free interest rate, we assume that positions are taken in futures
            with negligible margin requirements and that any cash is invested.

        sharpe_1d_lag:
            Sharpe for positions that are lagged by one day. A large difference between sharpe and
            1 day lagged Sharpe can indicate that the strategy has a significant short-term-alpha
            or that positions are taken with forward-looking-bias.

        sharpe_ex_slippage:
            Sharpe but without slippage. A large difference between sharpe and sharpe_ex_slippage
            indicates that the strategy will have more market impact when traded and will be harder
            to execute at size.

        sharpe_per_instrument:
            sharpe for each individual instrument.

        skewness:
            Daily skewness of the estimated total returns of the strategy (including slippage).

        slippages:
            Estimated cost of trading the strategy per instrument and day.

        rolling_position_bias:
            position_bias calculated on rolling time windows (1 year or as specified)

        rolling_sharpe:
            Sharpe calculated on rolling time windows (1 year or as specified)

        rolling_std:
            Standard deviation calculated on rolling windows (1 year or as specified)

    :param positions: A pandas dataframe of positons per date and instrument. Positions at time t are traded on the close at t.
    :param prices: A pandas dataframe of instrument prices per date.
    :param costs: A constant defining the slippage of a trade in daily standard deviations. Default -0.02
    :param rolling_window: The window length for rolling key figure calculations
    :param key_figures: A list of key figures to calculate. Valid key figure names are listed in ALL_KEY_FIGURES and in this docstring.
    :return: A dictionary containing key figures.
    """
    if key_figures is None:
        key_figures = DEFAULT_KEY_FIGURES

    if isinstance(positions, pd.Series):
        positions = positions.to_frame()

    positions = positions.shift(1)

    returns = prices.diff()
    volatility = (
        returns.rolling(window=100, min_periods=10).std().ffill().shift(1)
    )

    # Align indices and forward fill missing positions
    start = max([returns.index[0], positions.index[0]])
    end = min([returns.index[-1], positions.index[-1]])
    returns = returns.loc[start:end,]
    volatility = volatility.loc[start:end,]
    positions = positions.reindex(returns.index, method="ffill")

    # Precalculate common variables
    trades = positions.diff().abs()

    norm_positions = positions * volatility
    norm_trades = trades * volatility

    model_returns = returns * positions
    slippages = costs * norm_trades

    tot_returns = (model_returns + slippages).sum(axis=1, min_count=1)

    # Calculate key figures
    result = {}

    if "sharpe" in key_figures:
        result["sharpe"] = _calc_sharpe(tot_returns)

    if "sharpe_1d_lag" in key_figures:
        lagged_returns = returns * positions.shift(1) + costs * volatility * trades.shift(1)
        tot_lagged_returns = lagged_returns.sum(axis=1, min_count=1)
        result["sharpe_1d_lag"] = _calc_sharpe(tot_lagged_returns)

    if "sharpe_ex_slippage" in key_figures:
        result["sharpe_ex_slippage"] = _calc_sharpe(model_returns.sum(axis=1, min_count=1))

    if "sharpe_per_instrument" in key_figures:
        result["sharpe_per_instrument"] = _calc_sharpe(model_returns)

    if "av_holding" in key_figures:
        result["av_holding"] = (
            2
            * norm_positions.abs().sum(axis=1, min_count=1).mean()
            / norm_trades.abs().sum(axis=1, min_count=1).mean()
        )

    if "av_holding_per_instrument" in key_figures:
        result["av_holding_per_instrument"] = (
            2 * norm_positions.abs().mean() / norm_trades.abs().mean()
        )

    if "position_bias" in key_figures:
        result["position_bias"] = (
            norm_positions.sum(axis=1, min_count=1).mean()
            / norm_positions.abs().sum(axis=1, min_count=1).mean()
        )

    if "position_bias_per_instrument" in key_figures:
        result["position_bias_per_instrument"] = norm_positions.mean() / norm_positions.abs().mean()

    if "returns" in key_figures:
        result["returns"] = tot_returns

    if "slippages" in key_figures:
        result["slippages"] = slippages.sum(axis=1, min_count=1)

    if "skewness" in key_figures:
        result["skewness"] = tot_returns.skew()

    if "kurtosis" in key_figures:
        result["kurtosis"] = tot_returns.kurt() + 3

    if "rolling_std" in key_figures:
        result["rolling_std"] = tot_returns.rolling(
            rolling_window, min_periods=rolling_window // 2
        ).std()

    if "rolling_sharpe" in key_figures:
        result["rolling_sharpe"] = (
            tot_returns.rolling(rolling_window, min_periods=rolling_window // 2).mean()
            / tot_returns.rolling(rolling_window, min_periods=rolling_window // 2).std()
            * np.sqrt(260)
        )

    if "rolling_position_bias" in key_figures:
        long_positions = norm_positions.copy()
        long_positions[long_positions < 0] = 0
        result["rolling_position_bias"] = (
            (
                2 * long_positions.sum(axis=1, min_count=1)
                / norm_positions.abs().sum(axis=1, min_count=1)
                - 1
            )
            .rolling(180, min_periods=rolling_window // 2)
            .mean()
        )

    return result


def plot_key_figures(
    positions: pd.DataFrame, prices: pd.DataFrame, costs=-0.02, rolling_window=260
):
    """
    Calculates and plots key figures for strategy positions. Check calc_key_figures doc
    for more information about key figure definitions.

    :param positions: A pandas dataframe of positons per date and instrument. Positions at time t are traded on the close at t.
    :param prices: A pandas dataframe of instrument prices per date.
    :param costs: A constant defining the slippage of a trade (in daily standard deviations). Default 0.02
    :param rolling_window: The window length for rolling key figure calculations
    """
    key_figures = calc_key_figures(
        positions, prices, costs=costs, rolling_window=rolling_window, key_figures=ALL_KEY_FIGURES
    )

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    _plot_cumulative(ax0, key_figures, positions.index[[0, -1]])

    ax1 = fig.add_subplot(gs[0, 1])
    _plot_kf_table(ax1, key_figures)

    _plot_rolling(fig, gs[1, 0], key_figures, positions.index[[0, -1]], rolling_window)
    _plot_instrument_bars(fig, gs[1, 1], key_figures)


def _calc_sharpe(returns):
    return returns.mean() / returns.std() * np.sqrt(260)


def _plot_cumulative(ax0, key_figures, xlims):
    pd.concat([key_figures["returns"].cumsum(), key_figures["slippages"].cumsum()], axis=1).plot(
        ax=ax0
    )
    ax0.grid(zorder=-1.0)
    ax0.set_title("Cumulative return and slippage")
    plt.tick_params("x", labelbottom=True)
    plt.xlim(xlims)
    ax0.set_xlabel("")
    plt.legend(["Return inc. slippage", "Slippage"])
    ax0.set_ylabel("USD")


def _plot_instrument_bars(fig, gs, key_figures):
    gss11 = gs.subgridspec(3, 1, hspace=0.15)

    # Av. holding
    ax1 = fig.add_subplot(gss11[0, 0])
    key_figures["av_holding_per_instrument"].plot.bar(subplots=True, ax=ax1)
    plt.grid(axis="y", zorder=-1.0)
    ax1.set_ylabel("Av. holding")
    ax1.set_title("Per instrument")

    # Sharpe
    ax2 = fig.add_subplot(gss11[1, 0], sharex=ax1)
    key_figures["sharpe_per_instrument"].plot.bar(subplots=True, ax=ax2)
    plt.grid(axis="y", zorder=-1.0)
    ax2.set_ylabel("Sharpe")

    # Position bias
    ax3 = fig.add_subplot(gss11[2, 0], sharex=ax1)
    key_figures["position_bias_per_instrument"].plot.bar(subplots=True, ax=ax3)
    plt.grid(axis="y", zorder=-1.0)
    ax3.set_ylabel("Position bias")


def _plot_kf_table(ax1, key_figures):
    kfs = [
        ["Sharpe (yearly)", f'{key_figures["sharpe"]:.2f}'],
        ["Sharpe w. 1 day lagged pos. (yearly)", f'{key_figures["sharpe_1d_lag"]:.2f}'],
        ["Sharpe ex. slippage (yearly)", f'{key_figures["sharpe_ex_slippage"]:.2f}'],
        ["Skewness (daily)", f'{key_figures["skewness"]:.2f}'],
        ["Kurtosis (daily)", f'{key_figures["kurtosis"]:.2f}'],
        ["Average holding (in days)", f'{key_figures["av_holding"]:.2f}'],
        ["Position bias", f'{key_figures["position_bias"]:.2f}'],
    ]
    ax1.axis("off")
    tab1 = plt.table(cellText=kfs, loc="center", bbox=[0.4, 0.3, 0.3, 0.7], edges="open")
    tab1.auto_set_font_size(False)
    tab1.set_fontsize(11)


def _plot_rolling(fig, gs, key_figures, xlims, rolling_window):
    gss10 = gs.subgridspec(3, 1, hspace=0.15)

    # Rolling std
    ax1 = fig.add_subplot(gss10[0])
    key_figures["rolling_std"].plot(ax=ax1)
    ax1.set_ylabel("Standard dev.")
    plt.grid(zorder=-1.0)
    ax1.set_title(f"Rolling window ({rolling_window/260:.1g}y)")
    plt.tick_params("x", labelbottom=False)
    plt.xlim(xlims)

    # Rolling sharpe
    ax2 = fig.add_subplot(gss10[1], sharex=ax1)
    key_figures["rolling_sharpe"].plot(ax=ax2)
    ax2.set_ylabel("Sharpe")
    ax2.grid(zorder=-1.0)
    plt.tick_params("x", labelbottom=False)

    # Rolling lptp
    ax3 = fig.add_subplot(gss10[2], sharex=ax2)
    key_figures["rolling_position_bias"].plot(ax=ax3)
    ax3.set_ylabel("Position bias")
    ax3.grid(zorder=-1.0)
    ax3.set_xlabel("")
