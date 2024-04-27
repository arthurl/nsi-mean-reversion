# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,jp-MarkdownHeadingCollapsed,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Intraday Mean Reversion Trading on Indian Equity Markets

# %% [markdown]
# Import standard libraries

# %%
from IPython.core.display_functions import display
from types import NoneType
from collections.abc import Callable, Iterable, Iterator, Collection
from collections import namedtuple
import datetime
from pathlib import Path
import itertools
import functools
import sortedcontainers
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.pipeline
import sklearn.model_selection
import sklearn.utils

# %% [markdown]
# Flags / global config. Please modify them as needed.

# %%
BASEDIR: Path = Path(r"/home/arthur/Downloads/MScFE/capstone")
USELATEX = True
SKLCACHE = 4096  # cache size in MB to use with sklearn
RANDSEED = None
sns.set_context("paper")
sns.set_palette("deep")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Utilities and other python helper functions

# %% [markdown]
# Use latex for plotting to allow equations and to maintain the same typesetting
# as the paper. You will need to have latex installed.

# %%
if USELATEX:
    import re

    # from https://stackoverflow.com/a/25875504
    LATEXESCAPE = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    regex = re.compile(
        "|".join(
            re.escape(str(key))
            for key in sorted(LATEXESCAPE.keys(), key=lambda item: -len(item))
        )
    )

    def latexEscape(text: str) -> str:
        """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
        """
        return regex.sub(lambda match: LATEXESCAPE[match.group()], text)

    def latexTextSC(text: str) -> str:
        return r"\textsc{" + text + r"}"

    plt.rcParams.update(
        {"text.usetex": True, "font.family": "serif", "font.sans-serif": "XCharter"}
    )
else:

    def latexEscape(text: str) -> str:
        return text

    def latexTextSC(text: str) -> str:
        return text.upper()

# %% [markdown]
# Helper function for multiprocessing.


# %%
async def concurrent_map_fold(
    arr: Iterable, mapf: Callable, foldf: Callable, /, *, acc=None
):
    """Maps a function in parallel and folds the resulting sequence as and when
    they return."""
    import asyncio

    for coro in asyncio.as_completed([asyncio.to_thread(mapf, a) for a in arr]):
        res = await coro
        acc = res if acc is None else foldf(acc, res)
    return acc


# %% [markdown]
# Helper functions for time series processing.


# %%
def timeseriesAtTimes(
    ts: pd.Series | pd.DataFrame, dts: Iterable[datetime.time] | Iterable[str]
):
    return pd.concat(ts.at_time(t) for t in dts).sort_index()


def timeseriesByDay(ts: pd.Series | pd.DataFrame, dt: datetime.date):
    start = datetime.datetime.combine(dt, datetime.time())
    end = start + datetime.timedelta(days=1)
    return ts[(start <= ts.index) & (ts.index < end)]


def coalesceIntervals(intervals: Iterable[pd.Interval]) -> set[pd.Interval]:
    def joinOverlappingIntervals(intA: pd.Interval, intB: pd.Interval) -> pd.Interval:
        if intA.left < intB.left:
            leftTime = intA.left
            leftClosed = intA.closed_left
        elif intA.left == intB.left:
            leftTime = intA.left
            leftClosed = intA.closed_left or intB.closed_left
        else:
            leftTime = intB.left
            leftClosed = intB.closed_left
        if intA.right > intB.right:
            rightTime = intA.right
            rightClosed = intA.closed_right
        elif intA.right == intB.right:
            rightTime = intA.right
            rightClosed = intA.closed_right or intB.closed_right
        else:
            rightTime = intB.right
            rightClosed = intB.closed_right
        if leftClosed:
            closed = "both" if rightClosed else "left"
        else:
            closed = "right" if rightClosed else "neither"
        return pd.Interval(leftTime, rightTime, closed=closed)

    aggIntervals = set()
    for interval in intervals:
        overlapped = False
        for aggInterval in aggIntervals:
            if any(
                [
                    interval.overlaps(aggInterval),
                    interval.right == aggInterval.left
                    and (interval.closed_right or aggInterval.closed_left),
                    aggInterval.right == interval.left
                    and (aggInterval.closed_right or interval.closed_left),
                ]
            ):
                overlapped = True
                aggIntervals.remove(aggInterval)
                aggIntervals.add(joinOverlappingIntervals(interval, aggInterval))
                break
        if not overlapped:
            aggIntervals.add(interval)
    return aggIntervals


def getIntervalsWhereTrue(
    sr: pd.Series, /, *, key=lambda s: s.astype(bool), closed="right"
) -> pd.arrays.IntervalArray:
    sr = sr.sort_index()
    beforeTime = (
        sr.index
        - (sr.index - sr.index.to_series().shift(1, fill_value=sr.index[0])) / 2
    )
    afterTime = (
        sr.index
        + (sr.index.to_series().shift(-1, fill_value=sr.index[-1]) - sr.index) / 2
    )
    trueIntervals = coalesceIntervals(
        pd.arrays.IntervalArray.from_arrays(beforeTime, afterTime, closed=closed)[
            key(sr)
        ]
    )
    return pd.arrays.IntervalArray(data=np.array(sorted(trueIntervals)), closed=closed)


def batched(iterable: Iterable, n: int) -> Iterator:
    # from python docs
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


# %% [markdown]
# Time series indicators.


# %%
def logReturn(
    xs: pd.Series | pd.DataFrame, /, *, keepInitial=False
) -> pd.Series | pd.DataFrame:
    logRet = xs.pct_change().apply(np.log1p)
    return logRet if keepInitial else logRet.dropna()


def logReturnInv(
    xs: pd.Series | pd.DataFrame, /, *, fillInitialNaN=False
) -> pd.Series | pd.DataFrame:
    logRetInv = xs.cumsum().apply(np.exp)
    if fillInitialNaN:
        if np.isnan(logRetInv.iloc[0]):
            logRetInv.iloc[0] = 1.0
        else:
            raise ValueError(
                f"Initial value must be NaN if fillInitialNaN={fillInitialNaN}."
            )
    return logRetInv


def computeMACD(
    sr: pd.Series,
    /,
    *,
    short="12D",
    long="26D",
    ave="9D",
    center: bool | tuple[bool, bool, bool] = False,
):
    CMARK = "[c]"

    def ma(window, sr: pd.Series, center: bool) -> pd.Series:
        return sr.rolling(window=window, center=center).mean()

    cenShort, cenLong, cenAve = (
        (center, center, center) if isinstance(center, bool) else center
    )
    macd = ma(ave, ma(short, sr, cenShort) - ma(long, sr, cenLong), cenAve)
    macd.name = f"{sr.name} MACD({short}{CMARK if cenShort else ''}, {long}{CMARK if cenLong else ''}, {ave}{CMARK if cenAve else ''})"
    return macd


def computeRSI(sr: pd.Series, /, *, period: int | float = 14, minObservations=0):
    halflife = pd.Timedelta(days=-1 / np.log2(1 - 1 / period))
    delta = sr.diff()
    ewmParams = {
        "halflife": halflife,
        "times": delta.index,
        "min_periods": minObservations,
    }
    gain = delta.clip(lower=0).ewm(**ewmParams).mean()  # type: ignore
    loss = (-delta).clip(lower=0).ewm(**ewmParams).mean()  # type: ignore
    rsi = 100 - (100 / (1 + gain / loss))
    rsi.name = f"{sr.name} RSI({period}D)"
    return rsi


def computeBollingerBands(
    sr: pd.Series, /, *, period="20D", K: int | float = 2
) -> tuple[pd.DataFrame, pd.Series]:
    movingWin = sr.rolling(window=period, center=False)
    sma = movingWin.mean()
    stdDev = movingWin.std()
    stdDev.name = f"{sr.name} StdDev({period})"
    kStdDev = K * stdDev
    bollinger = pd.DataFrame()
    bollinger[f"{sr.name} Bollinger({period}, {K}) upper"] = sma + kStdDev
    bollinger[f"{sr.name} Bollinger({period}, {K}) lower"] = sma - kStdDev
    return bollinger, stdDev


# %% [markdown]
# Time series plotting functions.


# %%
def plotTimeseries(
    dataSets: Collection[
        pd.Series | pd.DataFrame
    ],  # first element uses the left y-axis, the others will use the right
    intervalSets: Collection[pd.arrays.IntervalArray] = [],
    bands: Collection[pd.DataFrame] = [],
    /,
    *,
    plotInterval: pd.Interval | NoneType = None,
    figsize=None,
    title: str = "",
    ylabels: Iterable[str] = iter([]),
    plotKWArgs: dict | Iterable[dict] | NoneType = None,
    intervalColours: Iterable[str] = iter([]),
    intervalKWArgs: dict | Iterable[dict] | NoneType = None,
    bandUpperColSuffix="upper",
    bandLowerColSuffix="lower",
    bandColours: Iterable[str] = iter([]),
    bandKWArgs: dict | Iterable[dict] | NoneType = None,
):
    defaultPlotKWArgs = {}
    defaultIntervalKWArgs = {"alpha": 0.2}
    defaultBandKWArgs = {"alpha": 0.1}

    def resolveKWArgs(
        length: int, kwArgs: dict | Iterable[dict] | NoneType, defaultKWArgs: dict
    ) -> list[dict]:
        kwArgs = defaultKWArgs if kwArgs is None else kwArgs
        if isinstance(kwArgs, dict):
            return [defaultKWArgs | kwArgs] * length
        else:
            result = [defaultKWArgs | args for args in kwArgs]
            if len(result) != length:
                raise ValueError(
                    "Length of keyword args must be the same as the input collection."
                )
            return result

    def colFromSuffix(df: pd.DataFrame, suffix: str) -> str:
        candidates = [col for col in df.columns if col.endswith(suffix)]
        if len(candidates) == 1:
            return candidates[0]
        else:
            raise ValueError(f"Cannot uniquely determine column from suffix {suffix}")

    if len(dataSets) == 0:
        return None
    plotInterval = plotInterval or pd.Interval(
        left=pd.Timestamp(datetime.datetime.min),
        right=pd.Timestamp(datetime.datetime.max),
        closed="left",
    )
    ylabelIt = iter(ylabels)
    plotKWArgsList = resolveKWArgs(len(dataSets), plotKWArgs, defaultPlotKWArgs)
    intervalColourIt = iter(intervalColours)
    intervalKWArgsList = resolveKWArgs(
        len(intervalSets), intervalKWArgs, defaultIntervalKWArgs
    )
    bandColourIt = iter(bandColours)
    bandKWArgsList = resolveKWArgs(len(bands), bandKWArgs, defaultBandKWArgs)

    fig, ax = plt.subplots(figsize=figsize)
    data: pd.DataFrame = next(dataSetIt := iter(dataSets))  # type: ignore
    cCounter = len(data.columns) if isinstance(data, pd.DataFrame) else 1
    colours = [f"C{i}" for i in range(cCounter)]
    data[(plotInterval.left <= data.index) & (data.index < plotInterval.right)].plot(
        ax=ax, legend=False, color=colours, **plotKWArgsList[0]
    )
    ax.set_xlabel("Date")
    if ylabel := next(ylabelIt, False):
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    axBBoxXMax = ax.get_tightbbox().xmax
    axPrev = ax

    for data, kwArgs in zip(dataSetIt, plotKWArgsList[1:], strict=True):
        axR = ax.twinx()
        # move spine out of the previous bounding box
        axR.spines.right.set_position(
            ("outward", axPrev.get_tightbbox().xmax - axBBoxXMax)
        )
        colours = [
            f"C{i}"
            for i in range(
                cCounter,
                cCounter + (len(data.columns) if isinstance(data, pd.DataFrame) else 1),
            )
        ]
        cCounter += len(colours)
        yAxisColour = colours[0] if len(dataSets) > 2 and len(colours) == 1 else None
        data[
            (plotInterval.left <= data.index) & (data.index < plotInterval.right)
        ].plot(ax=axR, legend=False, color=colours, **kwArgs)
        if ylabel := next(ylabelIt, False):
            if yAxisColour is None:
                axR.set_ylabel(ylabel)
            else:
                axR.set_ylabel(ylabel, color=yAxisColour)
        if yAxisColour is not None:
            axR.tick_params(axis="y", colors=yAxisColour)
        axPrev = axR

    for intervals, kwArgs in zip(intervalSets, intervalKWArgsList, strict=True):
        colour = next(intervalColourIt, None)
        if colour is None:
            colour = f"C{cCounter}"
            cCounter += 1
        for interval in intervals:
            if interval.overlaps(plotInterval):
                ax.axvspan(interval.left, interval.right, color=colour, **kwArgs)

    for band, kwArgs in zip(bands, bandKWArgsList, strict=True):
        colour = next(bandColourIt, None)
        if colour is None:
            colour = f"C{cCounter}"
            cCounter += 1
        band = band[
            (plotInterval.left <= band.index) & (band.index < plotInterval.right)
        ]
        lowerBand = band[colFromSuffix(band, bandLowerColSuffix)]
        upperBand = band[colFromSuffix(band, bandUpperColSuffix)]
        ax.fill_between(band.index, lowerBand, upperBand, color=colour, **kwArgs)

    ax.legend(handles=[ln for ax in fig.axes for ln in ax.get_lines()])
    return fig


# TODO: refactor to use plotTimeseries()
def plotIntraday(
    dataL: pd.Series | pd.DataFrame,
    dataR: pd.Series | pd.DataFrame | NoneType = None,
    /,
    *,
    dates: list[datetime.date] | NoneType = None,
    intradayFL: Callable = lambda x: x,
    intradayFR: Callable = lambda x: x,
    ncols=2,
    shareY=False,
    hideXTickLabels=False,
    legend: bool | str = "first",
    plotWidth=6.4,
    plotHeight=4.8,
    gridspecKW: dict | NoneType = None,
    yLabelpadR: float | NoneType = None,
    plotKWArgsL: dict | NoneType = None,
    plotKWArgsR: dict | NoneType = None,
):
    def renameDataframeOrSeries(data: pd.Series | pd.DataFrame, renameF: Callable):
        if isinstance(data, pd.DataFrame):
            data = data.rename(columns=renameF)
        else:
            data.name = renameF(data.name)
        return data

    dataL = renameDataframeOrSeries(dataL, latexEscape)
    if dataR is not None:
        dataR = renameDataframeOrSeries(dataR, latexEscape)
    if dates is None or not dates:
        raise NotImplementedError
    gridspecKW = gridspecKW or {}
    if shareY and "wspace" not in gridspecKW:
        gridspecKW["wspace"] = 0
    leftPlotKWArgs = plotKWArgsL or {}
    rightPlotKWArgs = plotKWArgsR or {}
    nrows = -(-len(dates) // ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=False,
        sharey=shareY,
        figsize=(plotWidth * ncols, plotHeight * nrows),
        gridspec_kw=gridspecKW,
    )
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]
    dtfmt = matplotlib.dates.DateFormatter("%H:%M")  # type: ignore
    for datesRow, axesRow in zip(batched(dates, ncols), axes, strict=True):
        for dt, ax in zip(datesRow, axesRow, strict=False):
            # x_compat needed for set_major_formatter see https://stackoverflow.com/a/70625107
            intradayFL(timeseriesByDay(dataL, dt)).plot(
                ax=ax, x_compat=True, legend=legend is not False, **leftPlotKWArgs
            )
            if dataR is not None:
                axR = intradayFR(timeseriesByDay(dataR, dt)).plot(
                    ax=ax,
                    x_compat=True,
                    legend=legend is not False,
                    secondary_y=True,
                    **rightPlotKWArgs,
                )
                axR.set_ylabel(axR.get_ylabel(), labelpad=yLabelpadR, rotation=270)
            ax.xaxis.set_major_formatter(dtfmt)
            ax.set_xlabel("")
            ax.set_title(dt)
            if legend == "first":
                legend = False
    if hideXTickLabels:
        for axesRow in axes:
            for ax in axesRow:
                ax.xaxis.set_ticklabels([])
    fig.tight_layout()
    return fig


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Bitcoin

# %% [markdown]
# Read raw tick-level data sourced from
# [https://bitcoincharts.com/about/markets-api]() in CSV format, and calculate a
# weighted-average price each second.

# %% tags=["active-py"]
data = pd.read_csv(
    BASEDIR / r"bitstampUSD.csv",
    dtype={"Time": int, "Price": float, "Amount": float},
)
data["Date"] = pd.to_datetime(data["Time"], unit="s")
del data["Time"]

EPSILON = 10**-9
weighted_mean = lambda x: np.average(x, weights=data.loc[x.index, "Amount"] + EPSILON)  # type: ignore
data = data.groupby("Date").agg(
    Price=("Price", weighted_mean), Amount=("Amount", "sum")
)
data.to_csv(BASEDIR / r"bitstampUSD.s.csv")

# %% [markdown]
# This process is slow, so we cache the results.

# %% tags=["active-py"]
data = pd.read_csv(
    BASEDIR / r"bitstampUSD.s.csv",
    index_col="Date",
    usecols=["Date", "Price", "Amount"],
    parse_dates=True,
)
display(data)

# %% [markdown]
# 30-minute time span plot of BTC price.

# %% tags=["active-py"]
fig, ax = plt.subplots()
data["Price"][
    (datetime.datetime(2023, 12, 31, 19, 0) < data.index)
    & (data.index < datetime.datetime(2023, 12, 31, 19, 30))
].plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel(latexEscape("Price / $"))

# %% [markdown]
# 1-month time span plot of BTC price.

# %% tags=["active-py"]
fig, ax = plt.subplots()
data["Price"][
    (datetime.datetime(2023, 12, 1, 0, 0) < data.index)
    & (data.index < datetime.datetime(2024, 1, 1, 0, 0))
].plot(ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel(latexEscape("Price / $"))

# %% [markdown]
# Forward fill price data to get a price each second.

# %% tags=["active-py"]
prices = data["Price"].copy()
prices.index = (data.index - data.index[0]) // pd.Timedelta("1s")
prices = prices.reindex(
    np.arange(prices.index[0], prices.index[-1]), fill_value=np.nan
).ffill()

fig, ax = plt.subplots()
prices.plot(ax=ax)
ax.set_xlabel(f"Seconds since {data.index[0]}")
ax.set_ylabel(latexEscape("Price / $"))

# %% [markdown]
# Find variance at different sampling frequencies.

# %% tags=["active-py"]
timescales = np.array([2**intervalpow for intervalpow in range(18)])
secondsInYear = 3600 * 24 * 365
annVariances = np.array(
    [prices[::ts].var() * secondsInYear / ts for ts in timescales]  # type: ignore
)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.plot(timescales, annVariances, "-o")
ax.set_xlabel("Time scale / seconds")
ax.set_ylabel("Annualised Variance")
fig.savefig(BASEDIR / r"M2 - BTC variance timescales.pdf", bbox_inches="tight")

# %% tags=["active-py"]
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(timescales, annVariances, "-o")
ax.set_xlabel("Time scale / seconds")
ax.set_ylabel("Annualised Variance")

# %% tags=["active-py"]
import sklearn.linear_model

reg = sklearn.linear_model.LinearRegression().fit(
    np.log(timescales).reshape(-1, 1), np.log(annVariances)
)
print(reg.intercept_, reg.coef_)

# %% tags=["active-py"]
np.array([prices[::ts].std() for ts in timescales])

# %% [markdown]
# Different approach: take rolling windows of equal size (and therefore
# different lengths of absolute time).


# %% tags=["active-py"]
def rollingStdScales(
    xs: Iterable, /, *, window: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    vmean, vstd = [], []
    arr = pd.Series(xs)  # type: ignore
    while len(arr) > window:
        vs = arr.rolling(window=window).std()
        vmean.append(vs.mean())
        vstd.append(vs.std())
        arr = arr[::2]
    return np.array(vmean), np.array(vstd)


stdM, stdStd = rollingStdScales(prices)
stdAnn = stdM * np.sqrt(secondsInYear / 2 ** np.arange(len(stdM)))

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(2 ** np.arange(len(stdAnn)), stdAnn, "-o")
ax.plot(2 ** np.arange(len(stdAnn)), stdAnn + stdStd, "-")
ax.plot(2 ** np.arange(len(stdAnn)), stdAnn - stdStd, "-")
ax.set_xlabel("Time scale / seconds")
ax.set_ylabel("Annualised Standard Deviation")

# %% tags=["active-py"]
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(2 ** np.arange(len(stdStd)), stdStd, "-o")
ax.set_xlabel("Time scale / seconds")
ax.set_ylabel("SD of SD")

# %% [markdown]
# ## Theory
#
# Visualisation of MACD filter Fourier coefficients. The array below was
# computed in Mathematica.

# %%
# fmt: off
macdFT = [
    0., 0., 0., 0., 0., 0.000546181, 0., 0.000936122, 0.000779668, 0., 0.00164285, 0.00280958, 0.00214851, 0.00274118, 0.0024081, 0.0027821, 0.00293688, 0.00183783, 0., 0.00173986, 0.00283944,
    0.00319732, 0.00309986, 0.0030794, 0.0035085, 0.00424001, 0.00496988, 0.00557641, 0.00616014, 0.00699306, 0.00837678, 0.0104667, 0.0132384, 0.0165777, 0.0203545, 0.0244503, 0.0287638, 0.0332096,
    0.0377167, 0.0422262, 0.0466901, 0.0510698, 0.0553345, 0.0594606, 0.0634302, 0.0672305, 0.0708527, 0.0742916, 0.0775447, 0.0806119, 0.0834949, 0.0861969, 0.0887222, 0.0910759, 0.0932639, 0.0952925,
    0.0971681, 0.0988975, 0.100488, 0.101945, 0.103277, 0.104489, 0.105588, 0.106581, 0.107473, 0.108271, 0.108978, 0.109602, 0.110147, 0.110618, 0.111019, 0.111355, 0.11163, 0.111847, 0.112011, 0.112124,
    0.112191, 0.112214, 0.112196, 0.11214, 0.112048, 0.111923, 0.111768, 0.111583, 0.111372, 0.111136, 0.110877, 0.110597, 0.110297, 0.109979, 0.109643, 0.109293, 0.108927, 0.108549, 0.108158, 0.107756,
    0.107344, 0.106922, 0.106492, 0.106055, 0.10561, 0.105159, 0.104702, 0.10424, 0.103774, 0.103304, 0.10283, 0.102353, 0.101874, 0.101392, 0.100909, 0.100424, 0.0999384, 0.0994517, 0.0989646, 0.0984772,
    0.0979899, 0.0975028, 0.0970162, 0.0965303, 0.0960453, 0.0955614, 0.0950787, 0.0945973, 0.0941175, 0.0936393, 0.093163, 0.0926885, 0.092216, 0.0917457, 0.0912775, 0.0908116, 0.090348, 0.0898869,
    0.0894283, 0.0889722, 0.0885187, 0.0880678, 0.0876196, 0.0871742, 0.0867315, 0.0862917, 0.0858546, 0.0854204, 0.0849891, 0.0845607, 0.0841351, 0.0837125, 0.0832928, 0.0828761, 0.0824622,
]
# fmt: on

fig, ax = plt.subplots()
ax.plot(macdFT)
ax.axvline(x=60, color="xkcd:pale red", alpha=0.7, linestyle="dotted")
ax.axvline(x=80, color="xkcd:pale red", alpha=0.7, linestyle="dotted")
ax.set_xlabel(r"Period, $\tau$ / days")
fig.savefig(
    BASEDIR / "M6 - MACD(24,52,18) fourier coefficients.pdf", bbox_inches="tight"
)
del macdFT

# %% [markdown]
# ## NSI Equity
#
# ### Metadata management

# %% [markdown]
# Helper functions to normalise tickers (by stripping spaces and capitalising)
# and parse date with time.


# %%
def normaliseTickers(tickers: pd.Series) -> pd.Series:
    import string

    return tickers.str.translate({ord(c): None for c in string.whitespace}).str.upper()


def cleanLabel(colname: str) -> str:
    return colname.strip().removeprefix("<").removesuffix(">")


def parseDateTime(
    df: pd.DataFrame,
    /,
    *,
    datecol: str,
    timecol: str,
    month: int,
    debugMsg: NoneType | str = None,
) -> pd.Series:
    import warnings

    def parseDate(dts, /, *, dayfirst: bool):
        try:
            parsedDates = pd.to_datetime(dts, dayfirst=dayfirst, yearfirst=False)
        except ValueError:
            parsedDates = None
        return parsedDates  # type: ignore

    # some files (e.g. PEL) have mixed date separators
    dateStrsClean = df[datecol].str.replace("/", "-")
    with warnings.catch_warnings(action="ignore"):
        # some file (e.g. PEL again) have mixed time formats. This is a horrible
        # hack to put them into a standard form.
        #
        # Dev note: ideally, we would use pd.to_timedelta, but that parser does
        # not recognise the hh:mm format (that yet again appears in e.g. PEL).
        timeStrsClean = pd.to_datetime(df[timecol]).apply(
            lambda x: x.strftime(r"%H:%M:%S")
        )
        datetimeStrsClean = dateStrsClean + " " + timeStrsClean
        # suppress warnings when dayfirst argument is ignored
        dayFirstDT = parseDate(datetimeStrsClean, dayfirst=True)
        monthFirstDT = parseDate(datetimeStrsClean, dayfirst=False)
    if dayFirstDT is not None and (
        monthFirstDT is None
        or (
            set(dayFirstDT.dt.month) == {month}
            and set(monthFirstDT.dt.month) != {month}
        )
    ):
        parsedDT = dayFirstDT
    elif monthFirstDT is not None and (
        dayFirstDT is None
        or (
            set(monthFirstDT.dt.month) == {month}
            and set(dayFirstDT.dt.month) != {month}
        )
    ):
        parsedDT = monthFirstDT
    elif dayFirstDT is not None and np.all(dayFirstDT == monthFirstDT):
        # sometimes exception will not be raised, and parsing quietly falls back
        parsedDT = dayFirstDT
    else:
        raise ValueError(
            "Ambiguous month" + (f": {debugMsg}" if debugMsg is not None else "")
        )
    # set month because some datasets (e.g. 5PAISA) have the wrong month
    parsedDT = parsedDT.apply(lambda d: d.replace(month=month))  # type: ignore
    parsedDT.name = "time"
    return parsedDT


# %% [markdown]
# Function to perform the heavy lifting of retrieving metadata from the RAR data
# files, such as a full list of tickers, and start/end dates + number of data
# points for each ticker.


# %%
async def fetchTickerInfo(
    *, basedir: Path = BASEDIR, startyear=2021, endyear=2023
) -> pd.DataFrame:
    import calendar
    from rarfile import RarFile

    def getStats(year: int, month: int) -> list[tuple[str, tuple]]:
        stats = []
        arxName = f"Cash Data {calendar.month_name[month]} {year}.rar"
        try:
            with RarFile(basedir / "data" / arxName, "r") as arx:
                dataPaths = [
                    path for path in map(Path, arx.namelist()) if path.suffix == ".csv"
                ]
                debugFileCount = 0
                for dataPath in dataPaths:
                    ticker = dataPath.stem
                    with arx.open(dataPath) as dataFile:
                        df = pd.read_csv(dataFile)
                    dates = parseDateTime(
                        df,
                        datecol="<date>",
                        timecol="<time>",
                        month=month,
                        debugMsg=f'in archive "{arxName}" file "{dataPath}" ',
                    )
                    startDate = min(dates)
                    endDate = max(dates)
                    count = len(df)
                    del df
                    stats.append((ticker, (startDate, endDate, count)))
                    debugFileCount += 1
                    if debugFileCount % 20 == 0:
                        print(".", end="")
        except FileNotFoundError:
            pass
        return stats

    def foldf(
        acc: dict[str, tuple], res: Iterable[tuple[str, tuple]]
    ) -> dict[str, tuple]:
        for ticker, (start, end, count) in res:
            if ticker in acc:
                s, e, c = acc[ticker]
                acc[ticker] = (
                    min(s, start),
                    max(e, end),
                    c + count,
                )
            else:
                acc[ticker] = (start, end, count)
        return acc

    tickerStats: dict[str, tuple] = await concurrent_map_fold(
        itertools.product(range(startyear, endyear + 1), range(1, 13)),
        lambda t: getStats(*t),
        foldf,
        acc={},
    )  # type: ignore
    tickers = list(tickerStats.keys())
    df = pd.DataFrame(
        {
            "start": [tickerStats[t][0] for t in tickers],
            "end": [tickerStats[t][1] for t in tickers],
            "count": [tickerStats[t][2] for t in tickers],
        },
        index=tickers,
    )
    df.index.name = "ticker"
    df.sort_index(inplace=True)
    return df


# %% tags=["active-py"]
info = await fetchTickerInfo()

# %% [markdown]
# Aside from the metadata that can be obtain from the data files, we also want
# to know the ticker type (e.g. equity / index / ETFs). To do this, we have to
# resort to manually fetching data from the Indian stock exchange.

# %% tags=["active-py"]
# from https://www.nseindia.com/market-data/securities-available-for-trading
with open(BASEDIR / "data" / "NSI" / "EQUITY_L.csv") as f:
    nsiEquities = set(normaliseTickers(pd.read_csv(f)["SYMBOL"]))
with open(BASEDIR / "data" / "NSI" / "eq_etfseclist.csv") as f:
    nsiETFs = set(normaliseTickers(pd.read_csv(f)["Symbol"]))
# fmt: off
# from https://upstox.com/stocks-market/share-market-listed-company-in-india/indices-index/
nsiIndices = (
    "ALLCAP", "AUTO", "BANKEX", "BHRT22", "BSE100", "BSE200", "BSE500", "BSECD", "BSECG", "BSEDSI", "BSEEVI", "BSEFMC", "BSEHC", "BSEIPO", "BSEIT", "BSELVI", "BSEMOI", "BSEPBI", "BSEPSU",
    "BSEQUI", "BSESER", "CARBON", "COMDTY", "CONDIS", "CPSE", "DFRGRI", "ENERGY", "ESG100", "FINSER", "GREENX", "INDSTR", "INFRA", "INDIA VIX", "LCTMCI", "LMI250", "LRGCAP", "METAL", "MFG", "MID150",
    "MIDCAP", "MIDSEL", "MSL400", "NIFTY ALPHA 50", "NIFTY ALPHALOWVOL", "NIFTY CONSR DURBL", "NIFTY HEALTHCARE", "NIFTY IND DIGITAL", "NIFTY INDIA MFG", "NIFTY LARGEMID250", "NIFTY M150 QLTY50",
    "NIFTY MICROCAP250", "NIFTY MID SELECT", "NIFTY MIDCAP 100", "NIFTY MIDCAP 150", "NIFTY MIDSML 400", "NIFTY OIL AND GAS", "NIFTY SMLCAP 100", "NIFTY SMLCAP 250", "NIFTY SMLCAP 50", "NIFTY TOTAL MKT",
    "NIFTY100 EQL WGT", "NIFTY100 ESG", "NIFTY100 LOWVOL30", "NIFTY100 QUALTY30", "NIFTY200 QUALTY30", "NIFTY50 EQL WGT", "NIFTY500 MULTICAP", "NIFTY 100", "NIFTY 200", "NIFTY 50", "NIFTY 500",
    "NIFTY AUTO", "NIFTY BANK", "NIFTY CPSE", "NIFTY COMMODITIES", "NIFTY CONSUMPTION", "NIFTY DIV OPPS 50", "NIFTY ENERGY", "NIFTY FMCG", "NIFTY FIN SERVICE", "NIFTY FINSRV25 50", "NIFTY GS 10YR",
    "NIFTY GS 10YR CLN", "NIFTY GS 11 15YR", "NIFTY GS 15YRPLUS", "NIFTY GS 4 8YR", "NIFTY GS 8 13YR", "NIFTY GS COMPSITE", "NIFTY GROWSECT 15", "NIFTY IT", "NIFTY INFRA", "NIFTY MNC", "NIFTY MEDIA",
    "NIFTY METAL", "NIFTY MID LIQ 15", "NIFTY MIDCAP 50", "NIFTY NEXT 50", "NIFTY PSE", "NIFTY PSU BANK", "NIFTY PHARMA", "NIFTY PVT BANK", "NIFTY REALTY", "NIFTY SERV SECTOR", "NIFTY100 LIQ 15",
    "NIFTY100ESGSECLDR", "NIFTY200 ALPHA 30", "NIFTY200MOMENTM30", "NIFTY50 DIV POINT", "NIFTY50 PR 1X INV", "NIFTY50 PR 2X LEV", "NIFTY50 TR 1X INV", "NIFTY50 TR 2X LEV", "NIFTY50 VALUE 20",
    "NIFTYM150MOMNTM50", "OILGAS", "POWER", "REALTY", "SENSEX", "SENSEX50", "SMEIPO", "SML250", "SMLCAP", "SMLSEL", "SNXT50", "TECK", "TELCOM", "UTILS",
)
# fmt: on
extraIndices = ("NIFTY100QLY30", "NIFTYSMALLCAP", "NIFTYMIDCAP", "NIFTYMID50")
nsiIndices = set(normaliseTickers(pd.Series(nsiIndices + extraIndices)))


def getTickerType(t: str) -> str:
    if t in nsiEquities:
        return "equity"
    elif t in nsiETFs:
        return "etf"
    elif t in nsiIndices:
        return "index"
    else:
        return ""


info["type"] = normaliseTickers(pd.Series(info.index, index=info.index)).apply(
    getTickerType
)

# %% [markdown]
# The `fetchTickerInfo` function is slow, as it has to dig through thousands of
# tickers. The following function fetches transaction volume and last price for
# a given month.


# %%
def fetchTradeVolumeAndPrice(
    year,
    month,
    /,
    *,
    basedir: Path = BASEDIR,
) -> pd.DataFrame:
    import calendar
    from rarfile import RarFile

    arxName = f"Cash Data {calendar.month_name[month]} {year}.rar"
    tickers = []
    volumePerDay = []
    lastPrice = []
    with RarFile(basedir / "data" / arxName, "r") as arx:
        dataPaths = [
            path for path in map(Path, arx.namelist()) if path.suffix == ".csv"
        ]
        for dataPath in dataPaths:
            ticker = dataPath.stem
            with arx.open(dataPath) as dataFile:
                df = pd.read_csv(dataFile)
            df.rename(columns=cleanLabel, inplace=True)
            df.index = parseDateTime(  # type: ignore
                df,
                datecol="date",
                timecol="time",
                month=month,
                debugMsg=f'in archive "{arxName}" file "{dataPath}" ',
            ).values
            tickers.append(ticker)
            volumePerDay.append(sum(df["volume"] * df["close"]) / len(df))
            lastPrice.append(df["close"].loc[max(df.index)])

    return pd.DataFrame(
        {
            f"{year}{month} $ volume per day": volumePerDay,
            f"{year}{month} last price": lastPrice,
        },
        index=tickers,
    )


# %% tags=["active-py"]
info = info.join(fetchTradeVolumeAndPrice(2023, 12), how="left")
info.to_csv(BASEDIR / "data" / "metadata.csv")

# %% [markdown]
# We cache the results in `metadata.csv` for convenience. If you don't have this
# file, you'll need to mark the cells above as "code" and run them.

# %%
with open(BASEDIR / "data" / "metadata.csv") as f:
    info = pd.read_csv(f, index_col="ticker", parse_dates=["start", "end"])
info["type"] = info["type"].fillna("").astype("string")
display(info)

# %% [markdown]
# Look at stock daily trading volumes.

# %%
volumes = info[
    (info["type"] == "equity")
    & (np.logical_not(np.isnan(info["202312 $ volume per day"])))
]["202312 $ volume per day"].values
volumes[::-1].sort()

fig, ax = plt.subplots()
ax.plot(np.arange(len(volumes)), volumes)
ax.set_yscale("log")
ax.set_xlabel("Rank")
ax.set_ylabel("Average daily volume in Dec 2023 / " + latexTextSC("inr"))
fig.savefig(BASEDIR / r"M4 - 202312 equity daily volumes.pdf", bbox_inches="tight")

# %% [markdown]
# We only want equity stocks that have good liquidity.

# %%
equityUniverse = info[
    (info["type"] == "equity")
    & (info["202312 $ volume per day"] > 5e6)
    & (info["202312 last price"] >= 100)
].sort_values("202312 $ volume per day", ascending=False)
display(equityUniverse.shape)
with pd.option_context("display.max_rows", 500):
    display(equityUniverse)

# %% [markdown]
# # Asset price analysis
#
# This function does the heavy lifting of retrieving a time series from the data
# files.


# %%
@functools.lru_cache(maxsize=32)
def fetchTicker(
    ticker: str,
    /,
    *,
    basedir: Path = BASEDIR,
    startyear=2021,
    endyear=2023,
) -> pd.DataFrame:
    import calendar
    from rarfile import RarFile

    data = []
    for year in range(startyear, endyear + 1):
        for month in range(1, 13):
            arxName = f"Cash Data {calendar.month_name[month]} {year}.rar"
            with RarFile(basedir / "data" / arxName, "r") as arx:
                indexArxPaths = [
                    path
                    for path in map(Path, arx.namelist())
                    if path.suffix == ".csv" and ticker == path.stem
                ]
                if len(indexArxPaths) > 1:
                    raise ValueError(
                        f"More than one file in archive {arxName} that matches ticker {ticker}"
                    )
                elif len(indexArxPaths) == 1:
                    with arx.open(indexArxPaths[0]) as f:
                        df = pd.read_csv(f)
                    df.rename(columns=cleanLabel, inplace=True)
                    df.index = parseDateTime(  # type: ignore
                        df,
                        datecol="date",
                        timecol="time",
                        month=month,
                        debugMsg=f"in archive {arxName}",
                    ).values
                    df.drop(
                        ["ticker", "date", "time"],
                        axis="columns",
                        inplace=True,
                        errors="ignore",
                    )
                    data.append(df)
    if not data:
        raise FileNotFoundError(f"Ticker {ticker} not found")
    df = pd.concat(data)
    allNullCols = df.isna().all(axis=0)
    df.drop(allNullCols[allNullCols].index, axis="columns", inplace=True)
    df.sort_index(inplace=True)
    df.index.name = "time"
    return df


def fetchTickersResampledAtTimes(
    tickers: Iterable[str],
    times: Iterable[str] | Iterable[datetime.time],
    /,
    *,
    colname="close",
) -> pd.DataFrame:
    def fillAsOf(sr: pd.Series, fullTickerData: dict[str, pd.Series]) -> pd.Series:
        naMask = sr.isna()
        sr.loc[naMask] = [
            fullTickerData[str(sr.name)].asof(t) for t in sr[naMask].index
        ]
        return sr

    times = list(times)  # type: ignore
    fullTickerData = {}
    resampledList = []
    for ticker in tickers:
        srFull = fetchTicker(ticker)[colname]
        srFull.name = ticker
        fullTickerData[ticker] = srFull
        sr = timeseriesAtTimes(srFull, times)
        sr.name = ticker
        resampledList.append(sr)
    resampledData = pd.DataFrame().join(resampledList, how="outer")
    resampledData.index.name = "time"
    resampledData.apply(fillAsOf, args=(fullTickerData,), axis=0)
    return resampledData


# %% [markdown]
# As an example, let’s look at the ticker “5PAISA”.

# %%
ticker = "5PAISA"
data = fetchTicker(ticker)
display(data)

# %%
fig, ax = plt.subplots()
data["close"][
    (datetime.datetime(2023, 1, 1, 0, 0) <= data.index)
    & (data.index < datetime.datetime(2024, 1, 1, 0, 0))
].plot(ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel(latexEscape("Price / $"))
fig.savefig(BASEDIR / r"M3 - 5PAISA 2023.pdf", bbox_inches="tight")

# %%
fig, ax = plt.subplots()
data["close"][
    (datetime.datetime(2023, 12, 29, 12, 30) <= data.index)
    & (data.index < datetime.datetime(2023, 12, 29, 14, 30))
].plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel(latexEscape("Price / $"))
fig.savefig(BASEDIR / r"M3 - 5PAISA 20231229 1230h.pdf", bbox_inches="tight")

# %% [markdown]
# IDEA on 3 Oct 2023. Note the quantised steps.

# %%
ticker = "IDEA"
data = fetchTicker(ticker)

fig = plotIntraday(
    data["close"],
    dates=[datetime.date(2023, 10, 3)],
    ncols=1,
    plotWidth=6.4,
    plotHeight=4.8,
)
fig.savefig(BASEDIR / r"M4 - IDEA 20231003.pdf", bbox_inches="tight")

# %% [markdown]
# GANGOTRI 2023 Q4 price. Note the illiquidity.

# %%
ticker = "GANGOTRI"
data = fetchTicker(ticker)

fig, ax = plt.subplots()
data["close"][
    (datetime.datetime(2023, 10, 1, 0, 0) <= data.index)
    & (data.index < datetime.datetime(2024, 1, 1, 0, 0))
].plot(ax=ax, style=".")
ax.set_xlabel("Date")
ax.set_ylabel(latexEscape("Price / $"))
fig.savefig(BASEDIR / r"M4 - GANGOTRI 2023Q4.pdf", bbox_inches="tight")

# %%
ticker = "HDFCBANK"
data = fetchTicker(ticker)

fig, ax = plt.subplots()
data["close"].plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel(latexEscape("Price / $"))
print(data.shape)


# %%
def findMaxReturnIntervals(
    prices: pd.Series, /, *, maxInterval: pd.Timedelta, queueSize=10, largest=True
) -> pd.DataFrame:
    def keepFirstN(d, queueSize: int) -> bool:
        hasDrop = False
        try:
            while True:
                d.popitem(index=queueSize)
                hasDrop = True
        except (IndexError, KeyError):
            pass
        return hasDrop

    returns = logReturn(prices, keepInitial=True)
    sortKeyF = (lambda x: (-x[0], x[1])) if largest else (lambda x: x)
    globalMaxSums = sortedcontainers.SortedDict(sortKeyF)
    maxSums = sortedcontainers.SortedDict(sortKeyF)
    hasDrop = False
    confidence = queueSize
    prevT, _ = next(rIt := iter(returns.items()))
    for t, val in rIt:
        for s, t0 in list(maxSums):
            del maxSums[(s, t0)]
            if t - t0 <= maxInterval:
                maxSums[(s + val, t0)] = t
        if hasDrop:
            confidence = min(confidence, len(maxSums))
        if t - prevT <= maxInterval:  # type: ignore
            maxSums[(val, prevT)] = t
        hasDrop = keepFirstN(maxSums, queueSize) or hasDrop
        globalMaxSums.update(maxSums.items())
        keepFirstN(globalMaxSums, queueSize)
        prevT = t
    df = pd.DataFrame.from_records(
        ((np.exp(s), t0, t1) for (s, t0), t1 in globalMaxSums.items()[:confidence]),
        columns=["return", "time start", "time end"],
    )
    df.index.name = "rank"
    return df


def findMaxReturnLongShortIntervals(
    prices: pd.Series, /, *, maxInterval: pd.Timedelta, queueSize=10
) -> pd.DataFrame:
    optimumLong = findMaxReturnIntervals(
        prices, maxInterval=maxInterval, queueSize=queueSize, largest=True
    )
    optimumLong["position"] = "long"
    optimumShort = findMaxReturnIntervals(
        prices, maxInterval=maxInterval, queueSize=queueSize, largest=False
    )
    optimumShort["return"] = 1 / optimumShort["return"]
    optimumShort["position"] = "short"
    optimum = (
        pd.concat([optimumLong, optimumShort])
        .sort_values(by="return", ascending=False)
        .head(n=min(len(optimumLong), len(optimumShort)))
        .reset_index(drop=True)
    )
    optimum.index.name = "rank"
    return optimum


# %% tags=["active-py"]
optimumTrades = findMaxReturnLongShortIntervals(
    data[data.index.minute == 30]["close"],  # type:ignore
    maxInterval=pd.Timedelta(days=21),
    queueSize=200,  # type: ignore
)

fig, ax = plt.subplots()
optimumTrades["return"].plot(ax=ax)
ax.set_xlabel("Rank")
ax.set_ylabel("Return")
ax.set_title(latexEscape(ticker))
# fig.savefig(BASEDIR / f"ranked return - {ticker}.png")
display(len(optimumTrades))

# %% tags=["active-py"]
start = datetime.datetime(2021, 1, 1, 0, 0)
end = start + pd.Timedelta(days=365)
df2 = pd.DataFrame()
for window in ["24h", "10D", "90D"]:
    df2[f"{window} MA"] = data["close"].rolling(window=window, center=True).mean()
df2 = df2[(start <= df2.index) & (df2.index < end)]

fig, ax = plt.subplots(figsize=(14, 10.5))
df2.plot(ax=ax)
for startTime, endTime in optimumTrades[
    (optimumTrades["time start"].between(start, end, inclusive="left"))
    & (optimumTrades["time end"].between(start, end, inclusive="left"))
][["time start", "time end"]].itertuples(index=False):
    ax.axvspan(startTime, endTime, alpha=0.2, color="xkcd:green")
ax.set_xlabel("Date")
ax.set_ylabel(latexEscape("Price / $"))
ax.set_title(latexEscape(ticker))
del df2


# %%
def findLargestIntervalsBefore0(
    sr: pd.Series,
    /,
    *,
    maxInterval: pd.Timedelta,
    threshold=0.5,  # number between 0 and 1. 0 means no filtering.
    thresholdInterval: pd.Timedelta | NoneType = None,
) -> pd.DataFrame:
    if thresholdInterval is None:
        thresholdInterval = 10 * maxInterval

    srSign: pd.Series = np.sign(sr)  # type: ignore
    intervals = (
        srSign[srSign.diff().fillna(0) != 0]
        .to_frame(name="direction")
        .reset_index(names="time end")
    )
    intervals["time start not before"] = (
        intervals["time end"] + pd.Timedelta(seconds=1)
    ).shift(1, fill_value=intervals["time end"].min() - maxInterval)
    intervals["time start not before"] = np.maximum(
        intervals["time start not before"], intervals["time end"] - maxInterval
    )

    def getIdxExtrema(row):
        subseries = sr[
            (row["time start not before"] <= sr.index) & (sr.index < row["time end"])
        ]
        return (
            (pd.Series.idxmin if row["direction"] > 0 else pd.Series.idxmax)(subseries)
            if len(subseries) > 0
            else np.NaN
        )

    intervals["time start"] = intervals.apply(getIdxExtrema, axis=1)
    # drop intervals without data
    intervals.dropna(inplace=True)
    intervals["time interval"] = pd.arrays.IntervalArray.from_arrays(
        intervals["time start"], intervals["time end"], closed="left"
    )
    intervals = intervals[["time interval", "direction"]].sort_values(
        by="time interval"
    )

    if threshold > 0:
        srAbs = sr.abs()
        intervals = intervals.drop(
            intervals[
                intervals["time interval"].array.left.map(srAbs)  # type: ignore
                / intervals.apply(
                    lambda row: 1
                    / srAbs[
                        (
                            row["time interval"].right - thresholdInterval / 2
                            <= srAbs.index
                        )
                        & (
                            srAbs.index
                            < row["time interval"].right + thresholdInterval / 2
                        )
                    ].max(),
                    axis=1,
                )
                < threshold
            ].index
        ).reset_index(drop=True)
    return intervals


def findMACDOptimumReturnIntervals(
    price: pd.Series,
    /,
    *,
    macdParams: dict | NoneType = None,
    macdCenter: bool | tuple[bool, bool, bool] = (True, False, True),
    maxInterval: pd.Timedelta | NoneType = None,
    thresholdMACD=0.2,
    thresholdReturn=0.02,
):
    macdParams = macdParams or {"short": "24D", "long": "52D", "ave": "18D"}
    maxInterval = maxInterval or pd.Timedelta(days=24)
    macd = computeMACD(price, **macdParams, center=macdCenter)
    optimumTrades = findLargestIntervalsBefore0(
        macd, maxInterval=maxInterval, threshold=thresholdMACD
    )
    optimumTrades["duration"] = optimumTrades[
        "time interval"
    ].array.length / pd.Timedelta(days=1)  # type: ignore
    optimumTrades["return"] = (
        optimumTrades["time interval"].array.right.map(price)  # type: ignore
        / optimumTrades["time interval"].array.left.map(price)  # type: ignore
    ) ** optimumTrades["direction"] - 1
    optimumTrades = optimumTrades.drop(
        optimumTrades[optimumTrades["return"] <= thresholdReturn].index
    ).reset_index(drop=True)
    return optimumTrades, macd


# %%
sr = data["close"]
sma = pd.DataFrame()
for window in ["24h", "30D"]:
    sma[f"{latexEscape(window)} {latexTextSC('ma')}"] = sr.rolling(
        window=window, center=True
    ).mean()
optimumTrades, macdCen = findMACDOptimumReturnIntervals(sr)

display(optimumTrades)

start = pd.Timestamp(2022, 1, 1)
plotInterval = pd.Interval(start, start + pd.Timedelta(days=365))
macdCen.name = latexTextSC("macd") + latexEscape(
    macdCen.name.removeprefix(f"{sr.name} MACD")  # type: ignore
)
sr.name = latexTextSC(latexEscape(ticker.lower()))
fig = plotTimeseries(
    [sma, macdCen],
    [
        optimumTrades[optimumTrades["direction"] > 0]["time interval"],
        optimumTrades[optimumTrades["direction"] < 0]["time interval"],
    ],  # type: ignore
    plotInterval=plotInterval,
    figsize=(8, 4.8),  # (14, 10.5)
    ylabels=[latexEscape("Price / $"), latexTextSC("macd") + latexEscape(" / $")],
    intervalColours=["xkcd:green", "xkcd:pale red"],
)
fig.axes[1].axhline(y=0, alpha=0.2, color="xkcd:grey", linestyle="--")  # type: ignore
fig.savefig(  # type: ignore
    BASEDIR / f"M6 - {ticker} {plotInterval.left.year} trades.pdf", bbox_inches="tight"
)
del sr, sma, macdCen, optimumTrades

# %%
equityUniverse.drop(
    equityUniverse[equityUniverse["start"].dt.date > datetime.date(2021, 1, 1)].index,
    inplace=True,
)
display(equityUniverse)

# %% tags=["active-py"]
resampledData = fetchTickersResampledAtTimes(equityUniverse.index, ["10:00", "14:45"])
resampledData.to_csv(BASEDIR / "data" / r"equity_universe_resampled_close.csv")

# %%
with open(BASEDIR / "data" / r"equity_universe_resampled_close.csv") as f:
    resampledData = pd.read_csv(f, index_col="time", parse_dates=True)
resampledData.columns = map(latexEscape, resampledData.columns)  # type: ignore
display(resampledData)

# %%
display(resampledData[resampledData.isna().any(axis=1)])

# %%
InferenceData = namedtuple("InferenceData", ["X", "y"])

# %%
sr = data["close"]
sr.name = ticker
srDaily = resampledData[ticker].at_time(datetime.time(10, 0))
ewm = pd.DataFrame()
for window in ["24h", "30D"]:
    ewm[f"{latexEscape(window)} {latexTextSC('ema')}"] = sr.ewm(
        halflife=pd.Timedelta(window),
        times=sr.index,  # type: ignore
        min_periods=10,
    ).mean()
macd = computeMACD(sr)
rsi = computeRSI(srDaily, minObservations=10)
bollinger, stdDev = computeBollingerBands(srDaily)
targetTrades, _ = findMACDOptimumReturnIntervals(sr)

display(targetTrades)

macd.name = latexTextSC("macd") + latexEscape(
    macd.name.removeprefix(f"{sr.name} MACD")  # type: ignore
)
rsi.name = latexTextSC("rsi") + latexEscape(
    rsi.name.removeprefix(f"{sr.name} RSI")  # type: ignore
)
bollinger.columns = [col.removeprefix(f"{sr.name} ") for col in bollinger.columns]
stdDev.name = latexEscape(stdDev.name.removeprefix(f"{sr.name} "))  # type: ignore
sr.name = latexTextSC(latexEscape(ticker.lower()))
srDaily.name = sr.name
for yr in [2021, 2022, 2023]:
    plotInterval = pd.Interval(
        pd.Timestamp(yr, 1, 1), pd.Timestamp(yr + 1, 1, 1), closed="left"
    )
    fig = plotTimeseries(
        [ewm, macd, rsi],
        [
            targetTrades[targetTrades["direction"] > 0]["time interval"],
            targetTrades[targetTrades["direction"] < 0]["time interval"],
        ],  # type: ignore
        [bollinger],
        plotInterval=plotInterval,
        figsize=(14, 10.5),  # (8, 4.8),
        title=f"{sr.name} {plotInterval.left.year}",
        ylabels=[
            latexEscape("Price / $"),
            latexTextSC("macd") + latexEscape(" / $"),
            latexTextSC("rsi"),
        ],
        plotKWArgs={"alpha": 0.8},
        intervalColours=["xkcd:green", "xkcd:pale red"],
        bandColours=["C0"],
    )
    fig.axes[2].axhline(y=20, alpha=0.2, color="xkcd:grey", linestyle="--")  # type: ignore
    fig.axes[2].axhline(y=80, alpha=0.2, color="xkcd:grey", linestyle="--")  # type: ignore
    del yr, plotInterval


def getDirection(t) -> int:
    for interval, direction in targetTrades[["time interval", "direction"]].itertuples(
        index=False
    ):
        if t in interval:
            return direction
    return 0


rescaledFeatures = (
    stdDev.to_frame()
    .join([ewm, macd, srDaily.shift(range(1, 22))], how="left", sort=True)  # type: ignore
    .dropna()
    .div(srDaily, axis="index")
)
X = rescaledFeatures.join([rsi], how="left", sort=True).dropna()
y = pd.Series(data=X.index.map(getDirection), index=X.index)
XTrain, XTest, yTrain, yTest = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, shuffle=False
)
infData = {"train": InferenceData(XTrain, yTrain), "test": InferenceData(XTest, yTest)}
infData = {
    k: InferenceData(*sklearn.utils.shuffle(*v, random_state=RANDSEED))
    for k, v in infData.items()
}
del rescaledFeatures, X, y, XTrain, XTest, yTrain, yTest
del sr, srDaily, rsi, stdDev, _

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

balanceInvested = {0: sum(infData["train"].y != 0) / len(infData["train"].y)}
balanceInvested[1] = (1 - balanceInvested[0]) / 2
balanceInvested[-1] = balanceInvested[1]
paramSpace = {
    "svc__C": [10**i for i in range(-9, 9)],
    "svc__kernel": ["rbf"],
    "svc__class_weight": ["balanced", balanceInvested],
}
pipe = sklearn.pipeline.make_pipeline(
    StandardScaler(),
    SVC(probability=True, break_ties=True, cache_size=SKLCACHE, random_state=RANDSEED),
)
clf = sklearn.model_selection.GridSearchCV(pipe, paramSpace, n_jobs=-2, verbose=0)
del balanceInvested, paramSpace, pipe
clf.fit(*infData["train"])
display(clf.best_params_)

# %%
for label, (X, y) in infData.items():
    yPred = pd.Series(data=clf.predict(X), index=X.index)  # type: ignore
    fig = plotTimeseries(
        [ewm, macd],
        [
            targetTrades[targetTrades["direction"] > 0]["time interval"],  # type: ignore
            targetTrades[targetTrades["direction"] < 0]["time interval"],  # type: ignore
            getIntervalsWhereTrue(yPred, key=lambda x: x > 0),
            getIntervalsWhereTrue(yPred, key=lambda x: x < 0),
        ],
        [bollinger],
        plotInterval=pd.Interval(X.index.min(), X.index.max()),
        figsize=(14, 10.5),  # (8, 4.8),
        title=f"{latexTextSC(latexEscape(ticker.lower()))} ({label})",
        ylabels=[
            latexEscape("Price / $"),
            latexTextSC("macd") + latexEscape(" / $"),
        ],
        plotKWArgs={"alpha": 0.8},
        intervalColours=["xkcd:green", "xkcd:pale red"] * 2,
        intervalKWArgs=[
            {"ymin": 0, "ymax": 0.7},
            {"ymin": 0.01, "ymax": 0.71},
            {"ymin": 0.29, "ymax": 0.99},
            {"ymin": 0.3, "ymax": 1},
        ],
        bandColours=["C0"],
    )
    axes: list = fig.axes  # type: ignore
    axes[1].axhline(y=0, alpha=0.2, color="xkcd:grey", linestyle="--")
    axes[0].annotate("(predicted intervals)", (0.1, 0.9), xycoords="axes fraction")
    axes[0].annotate("(target intervals)", (0.1, 0.1), xycoords="axes fraction")
    del label, X, y, yPred, axes

# %%
import sklearn.metrics

fig, ax = plt.subplots()
X, y = infData["test"]
yProb = clf.predict_proba(X)[:, 1]  # type:ignore
fpr, tpr, threshold = sklearn.metrics.roc_curve(y, yProb)
lines = ax.plot(fpr, threshold, color="C0", label="Threshold (test)")
ax.set_xlabel(latexTextSC("fpr"))
ax.set_ylabel("Threshold")
axR = ax.twinx()
for i, (label, (X, y)) in enumerate(infData.items(), start=1):
    yProb = clf.predict_proba(X)[:, 1]  # type:ignore
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y, yProb)
    lines.extend(
        axR.plot(fpr, tpr, color=f"C{i}", label=f"{latexTextSC('roc')} ({label})")
    )
    print(f"{label} roc_auc_score = {sklearn.metrics.roc_auc_score(y, yProb)}")
    del i, label
axR.plot(fpr, fpr, alpha=0.2, color="xkcd:grey", linestyle="--")
axR.set_ylabel(latexTextSC("tpr"))
ax.legend(handles=lines, loc="center right")
del fig, ax, X, y, yProb, fpr, tpr, threshold, lines


# %% [markdown]
# Next example, “SUNPHARMA”.

# %%
ticker = "SUNPHARMA"
data = fetchTicker(ticker)

fig, ax = plt.subplots()
data["close"].plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel(latexEscape("Price / $"))
print(data.shape)

# %%
ticker = "RELIANCE"
data = fetchTicker(ticker)

fig, ax = plt.subplots()
data["close"].plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel(latexEscape("Price / $"))
print(data.shape)

# %%
ticker = "SBIN"
data = fetchTicker(ticker)

fig, ax = plt.subplots()
data["close"].plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel(latexEscape("Price / $"))
print(data.shape)

# %%
ticker = "ADANIPORTS"
data = fetchTicker(ticker)

fig, ax = plt.subplots()
data["close"].plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel(latexEscape("Price / $"))
print(data.shape)

# %%
ticker = "NIFTYBEES"
data = fetchTicker(ticker)

fig, ax = plt.subplots()
data["close"].plot(ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel(latexEscape("Price / $"))
print(data.shape)

# %%
ticker = "INDIAVIX"
data = fetchTicker(ticker)
data.to_csv(BASEDIR / r"data.csv")

fig, ax = plt.subplots()
data["close"][
    (datetime.datetime(2021, 1, 1, 0, 0) <= data.index)
    & (data.index < datetime.datetime(2021, 4, 1, 0, 0))
].plot(ax=ax, style=".")
ax.set_xlabel("Time")
ax.set_ylabel(latexEscape("Price / $"))
print(data.shape)
