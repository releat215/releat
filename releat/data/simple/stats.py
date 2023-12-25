"""Statistical Features.

Simple statistical calculations for tick groups


"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import polars as pl
import scipy.signal as signal
from numba import njit
from p_tqdm import p_map
from scipy import optimize
from scipy.ndimage import gaussian_filter1d


def get_last(df_group, fc):
    """Last value in group.

    Args:
        df_group (pl.GroupBy):
            ticks grouped by the feature timeframe
        fc (pydantic.BaseModel):
            feature config

    Returns:
        pl.DataFrame:
            last value for each group

    """
    col = fc.inputs[0]
    df = df_group.agg(
        [
            pl.col(col).last().alias("feat"),
        ],
    )
    return df


def get_mean(df_group, fc):
    """Average value in group.

    Args:
        df_group (pl.GroupBy):
            ticks grouped by the feature timeframe
        fc (pydantic.BaseModel):
            feature config

    Returns:
        pl.DataFrame:
            average value for each group

    """
    col = fc.inputs[0]
    df = df_group.agg(
        [
            pl.col(col).mean().alias("feat"),
        ],
    )
    return df


def get_min(df_group, fc):
    """Min value in group.

    Args:
        df_group (pl.GroupBy):
            ticks grouped by the feature timeframe
        fc (pydantic.BaseModel):
            feature config

    Returns:
        pl.DataFrame:
            min value for each group relative to
            the average value of the group

    """
    col = fc.inputs[0]
    df = df_group.agg(
        [
            (pl.col(col).mean() - pl.col(col).min()).alias("feat"),
        ],
    )
    return df


def get_max(df_group, fc):
    """Max value in group.

    Args:
        df_group (pl.GroupBy):
            ticks grouped by the feature timeframe
        fc (pydantic.BaseModel):
            feature config

    Returns:
        pl.DataFrame:
            max value for each group relative to
            the average value of the group

    """
    col = fc.inputs[0]
    df = df_group.agg(
        [
            (pl.col(col).max() - pl.col(col).mean()).alias("feat"),
        ],
    )
    return df


def get_skew(df_group, fc, pip):
    """Skew of group.

    Args:
        df_group (pl.GroupBy):
            ticks grouped by the feature timeframe
        fc (pydantic.BaseModel):
            feature config
        pip (float):
            pip value, i.e. 1e-4 for EURUSD

    Returns:
        pl.DataFrame:
            skew of group if there are more than min_num tick in group.
            #TODO scaling doesnt affect skew, remove pip input

    """
    col = fc.inputs[0]
    df = (
        df_group.agg(
            [
                ((pl.col(col) - pl.col(col).first()) / pip).skew().alias("feat"),
                pl.col(col).alias("raw"),
            ],
        )
        .with_columns(
            pl.when(pl.col("raw").list.lengths() > fc.kwargs["min_num"])
            .then(pl.col("feat"))
            .otherwise(0)
            .alias("feat"),
        )
        .select(["time_msc", "feat"])
    )
    return df


def one_hot_fx_flag(df_group, fc):
    """One hot enconding for flag.

    Applies to fx only because futures and stocks have more flags
    #TODO create a one_hot_flag for other trading instruments

    Args:
        df_group (pl.GroupBy):
            ticks grouped by the feature timeframe
        fc (pydantic.BaseModel):
            feature config

    Returns:
        pl.DataFrame:
            one hot encoded flags where the values are the total number
            of flags + percentage made up by 2 and 4 flags.

    """
    col = fc.inputs[0]
    df = (
        df_group.agg(
            [
                (pl.col(col) == 2).sum().alias("feat_2"),
                (pl.col(col) == 4).sum().alias("feat_4"),
                (pl.col(col) == 6).sum().alias("feat_6"),
            ],
        )
        .with_columns(
            (pl.col("feat_6") + pl.col("feat_4") + pl.col("feat_2"))
            .cast(pl.Int32)
            .alias("feat_6"),
        )
        .with_columns(
            (pl.col("feat_2") / pl.col("feat_6")).cast(pl.Float32).alias("feat_2"),
        )
        .with_columns(
            (pl.col("feat_4") / pl.col("feat_6")).cast(pl.Float32).alias("feat_4"),
        )
    )
    return df


@njit(
    "Tuple((float32[:],float32[:,:],float32[:],float32[:]))(float32[:,:], float32[:,:])",
    cache=True, nogil=True, fastmath=True
)
def calc_grad_and_error(x, y):
    """Calc grad and error.

    Simple linear least squares

    Args:
        x (np.array)
            1-d array (i think)
        y (np.array)
            also 1-d i think

    Returns:
        grad (np.float)
            gradient in radians
        err (np.float)
            error
        m (np.float)
            gradient as integer on euclidean plane
        c (np.float)
            y intercept

    """
    A = np.hstack((x, np.ones((len(x), 1), dtype="float32")))
    m, c = np.linalg.lstsq(A, y)[0]

    # main gradient
    grad = np.arctan(m) * 180 / math.pi

    # calculate residuals
    err = y - (m * x[:, :1] + c)
    return grad.astype("float32"), err, m, c


@njit("float32(float32[:],float32[:],float32,int32)",cache=True, nogil=True, fastmath=True)
def calc_grad(x, y, pip=1e-4, min_num=10):
    """Calc gradient feature.

    Wrapper for calc_grad_and_error

    Args:
        x (np.array)
            1 d array
        y (np.array)
            1 d i think
        pip (float)
            pip value
        min_num (int)
            minimum required to return gradient

    Returns:
        np.float
            gradient in radians

    """
    if len(y) < min_num:
        return 0.0

    # y = y * (len(y) / (y.max() - y.min()))
    y = (y - y.min()) / pip
    y = y.astype("float32").reshape((-1, 1))
    # x = np.arange(len(y)).astype("float32").reshape(-1, 1)
    x = x.astype("float32").reshape((-1, 1))
    grad, _, _, _ = calc_grad_and_error(x, y)
    return grad[0]


def calc_gradient_feature(df_group, fc, pip):
    """Apply gradient to group.

    Args:
        df_group (pl.GroupBy):
            ticks grouped by the feature timeframe
        fc (dict):
            feature configuration
        pip (float):
            i.e. 1e-4

    Returns:
        pl.DataFrame:
            Gradient for tick group if there are more than min_num ticks

    """
    col = fc.inputs[0]

    def get_grad(pip: float, min_num: int, struct: dict) -> pl.Series:
        """Gradient of series."""
        x = np.array(struct["x"], dtype=np.float32)
        y = np.array(struct["y"], dtype=np.float32)

        if len(y) > 10:
            try:
                val = calc_grad(x, y, pip, min_num)
                return tuple([val])
            except Exception:
                return tuple([0.0])

    p_get_grad = partial(get_grad, pip, fc.kwargs["min_num"])

    df = (
        df_group.agg(
            [
                pl.col(col).alias("y"),
                (
                    (pl.col("time_msc") - pl.col("time_msc").first()).cast(pl.Int64)
                    * 1e-9
                    / 60
                ).alias("x"),
            ],
        )
        .with_columns(pl.struct(["x", "y"]).apply(p_get_grad).alias("feat"))
        .with_columns(pl.col("feat").cast(pl.List(pl.Float32)))
        .with_columns(pl.col("feat").list.get(0))
        .select(["time_msc", "feat"])
        .lazy()
        .collect(streaming=True)
    )

    return df


def calc_grad_and_peak_trends(x, y, pip=1e-4, fs_factor=10, w1=0.2, w2=2.5, min_num=500):
    """Calc gradient and peak trends.

    Calculates the gradient of the peaks and gradient of the troughs, relative
    to the overall gradient.

    Args:
        y (np.array)
            1-d array (I think)
        fs_factor (int)
        w1 (np.float)
        w2 (np.float)
        min_num (int)

    Returns:
        grad (np.float)
            gradient in radians of all data
        t_grad (np.float)
            difference between grad and gradient of troughs
        p_grad (np.float)
            difference between grad and gradient of peaks


    """
    # if insufficient samples, then 0 gradient for all
    if len(y) < min_num:
        return tuple([0.0, 0.0, 0.0])

    y = (y - y.min()) / pip
    y = y.astype("float32")

    fs = len(y) // fs_factor

    # find peak and troughs using irr filter
    sos = signal.iirfilter(
        2,
        Wn=[w1, w2],
        fs=fs,
        btype="bandpass",
        ftype="butter",
        output="sos",
    )
    yfilt = signal.sosfiltfilt(sos, y)
    tMax, _ = signal.find_peaks(yfilt, distance=0.35 * fs, height=0.0)
    tMin, _ = signal.find_peaks(-yfilt, distance=0.35 * fs, height=0.0)

    # gradient of peaks
    p_grad, _, _, _ = calc_grad_and_error(
        x[tMax].astype("float32").reshape(-1, 1),
        y[tMax].astype("float32").reshape(-1, 1),
    )

    # gradient of trough
    t_grad, _, _, _ = calc_grad_and_error(
        x[tMin].astype("float32").reshape(-1, 1),
        y[tMin].astype("float32").reshape(-1, 1),
    )

    # overall gradient
    grad, _, _, _ = calc_grad_and_error(
        x.astype("float32").reshape(-1, 1),
        y.astype("float32").reshape(-1, 1),
    )

    return tuple([grad[0], t_grad[0] - grad[0], p_grad[0] - grad[0]])


def calc_peak_trough_gradient_feature(df_group, fc, pip):
    """Calculate peak and trough.

    TODO clean up hacky batch logic - the apply function polars groups is single
    threaded making it much faster to convert to numpy, build feature using
    multiprocessing and then convert back to polars

    Args:
        df_group (pl.GroupBy)
            polars group by dataframe, where the tick data is grouped by the feature
            timeframe
        fc (pydantic.BaseModel):
            feature config
        pip (float):
            value of one pip

    Returns:
        pl.DataFrame
            a feature (3 outputs) for each trade timeframe

    """
    col = fc.inputs[0]

    args = fc.kwargs

    df = df_group.agg(
        [
            pl.col(col).alias("y") * 10,
            (
                (pl.col("time_msc") - pl.col("time_msc").first()).cast(pl.Int64) * 1e-9
            ).alias("x"),
        ],
    )

    # if running a small sample or at inference, performance is negligble
    if len(df) < 20_000:

        def get_peak_trough(
            pip: float,
            fs_factor: int,
            w1: float,
            w2: float,
            min_num: int,
            struct,
        ):
            """Calculate trough and peak."""
            x = np.array(struct["x"], dtype=np.float32)
            y = np.array(struct["y"], dtype=np.float32)
            val = calc_grad_and_peak_trends(x, y, pip, fs_factor, w1, w2, min_num)
            return pl.Series(val, dtype=pl.Float32)

        p_get_peak_trough = partial(
            get_peak_trough,
            pip,
            args["fs_factor"],
            args["w1"],
            args["w2"],
            args["min_num"],
        )
        df = df.with_columns(
            pl.struct(["x", "y"]).apply(p_get_peak_trough).alias("feat"),
        )

    # for large tick data sets, faster to run using multiple processes
    else:
        df_np = df.to_numpy()

        def get_peak_trough(
            pip: float,
            fs_factor: int,
            w1: float,
            w2: float,
            min_num: int,
            vals,
        ):
            """Calculate trough and peak."""
            x = vals[2]
            y = vals[1]
            val = calc_grad_and_peak_trends(x, y, pip, fs_factor, w1, w2, min_num)
            return val

        p_get_peak_trough = partial(
            get_peak_trough,
            pip,
            args["fs_factor"],
            args["w1"],
            args["w2"],
            args["min_num"],
        )
        df_np = p_map(p_get_peak_trough, df_np, num_cpus=13)
        df = df.with_columns(pl.Series(df_np).alias("feat"))

    df = (
        df.select(["time_msc", "feat"])
        .with_columns(pl.col("feat").cast(pl.List(pl.Float32)))
        .with_columns(pl.col("feat").list.to_struct())
        .unnest("feat")
    )

    return df


def guess_initial_sine_param(tt, yy):
    """Initial guess for curve fitting.

    Fit sine to the input time sequence

    Args:
        tt (np.array):
            1d x co-ordinates for curve fitting - in this case either index of
            the tick or milliseconds between ticks
        yy (np.array):
            1d y co-ordinates for curve fitting - in this case the scaled value
            of the tick

    Returns:
        np.array:
            1d array for estimating initial values of amplitude, omega, phase and
            offset

    """
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(
        ff[np.argmax(Fyy[1:]) + 1],
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.0**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])
    return guess


def get_inflection_point(x, y, pip=1e-4, min_num=5, sigma=20):
    """Get inflection.

    Args:
        x (np.array):
            1d array of data, usually index of tick or milliseconds between ticks
        y (np.array):
            1d array of tick values
        pip (float):
            pip value, i.e. 1e-4
        min_num (int):
            minimum number of ticks required to calculate the inflection
        sigma (int):
            controls spread around the mean of the gaussian filter

    """
    # if too few ticks, return 0
    if len(y) < min_num:
        return np.zeros((4,), dtype="float32")

    y = (y - y.min()) / pip

    nx = np.arange(0, x.max(), 1e-2)  # choose new x axis sampling
    ny = np.interp(nx, x, y)  # generate y values for each xtick_df

    smooth = gaussian_filter1d(ny, sigma=sigma)

    # compute second derivative
    smooth_d2 = np.gradient(np.gradient(smooth))

    # find switching points
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]

    if len(infls) == 0:
        return np.zeros((4,), dtype="float32")

    infl_ind = infls[-1]
    half_width = len(nx) - infl_ind

    if half_width < min_num:
        return np.zeros((4,), dtype="float32")

    start_ind = max(len(nx) - 2 * half_width, 0)
    x1 = nx[start_ind:] - nx[infl_ind]
    y1 = smooth[start_ind:] - smooth[infl_ind]

    # z = np.polyfit(x1,y1,3)
    # z = Chebyshev.fit(x1,y1,3)
    def sine_func(x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    guess = guess_initial_sine_param(x1, y1)
    try:
        z, _ = optimize.curve_fit(sine_func, x1, y1, p0=guess)
        # z = np.polynomial.chebyshev.chebfit(x1,y1,3)
        z.astype("float32")

        return z
    except RuntimeError as e:
        _ = str(repr(e))
        return np.zeros((4,), dtype="float32")


def calc_inflection_feature(df_group, fc, pip):
    """Calculate inflection.

    TODO clean up hacky batch logic - the apply function polars groups is single
    threaded making it much faster to convert to numpy, build feature using
    multiprocessing and then convert back to polars

    Args:
        df_group (pl.GroupBy)
            polars group by dataframe, where the tick data is grouped by the feature
            timeframe
        fc (pydantic.BaseModel):
            feature config
        pip (float):
            value of one pip

    Returns:
        pl.DataFrame
            a feature (3 outputs) for each trade timeframe

    """
    col = fc.inputs[0]

    args = fc.kwargs

    df = df_group.agg(
        [
            pl.col(col).alias("y") * 10,
            (
                (pl.col("time_msc") - pl.col("time_msc").first()).cast(pl.Int64) * 1e-9
            ).alias("x"),
        ],
    )

    if len(df) < 20_000:

        def get_inflection(
            pip: float,
            min_num: int,
            sigma: int,
            struct: dict,
        ) -> pl.Series:
            """Get inflection."""
            x = np.array(struct["x"], dtype=np.float32)
            y = np.array(struct["y"], dtype=np.float32)
            val = get_inflection_point(x, y, pip, min_num, sigma)
            return pl.Series(val, dtype=pl.Float32)

        p_get_inflection = partial(get_inflection, pip, args["min_num"], args["sigma"])

        df = df.with_columns(pl.struct(["x", "y"]).apply(p_get_inflection).alias("feat"))

    else:
        df_np = df.to_numpy()

        def get_inflection(pip: float, min_num: int, sigma: int, vals):
            """Calculate inflection."""
            x = vals[2]
            y = vals[1]
            val = get_inflection_point(x, y, pip, min_num, sigma)
            return val.tolist()

        p_get_inflection = partial(get_inflection, pip, args["min_num"], args["sigma"])
        df_np = p_map(p_get_inflection, df_np, num_cpus=13)
        df = df.with_columns(pl.Series(df_np).alias("feat"))

    df = (
        df.select(["time_msc", "feat"])
        .with_columns(pl.col("feat").list.to_struct())
        .unnest("feat")
    )

    return df


@njit("int32(int32, int32)",cache=True, nogil=True, fastmath=True)
def randint(low, high):
    """Random integer.

    Args:
        low (int)
        high (int)

    Returns:
        int
            an integer between the low and high numbers

    """
    return np.random.randint(low, high)


@njit("float32[:](float32[:])",cache=True, nogil=True, fastmath=True)
def sign(ts):
    """Sign.

    Args:
        ts (float or np.array)

    Returns:
        np.array

    """
    return np.sign(ts)


@njit("float32[:](float32[:],int32)",cache=True, nogil=True, fastmath=True)
def log(ts, base=-1):
    """Log.

    Args:
        ts (float or np.array)
        base (int)

    Returns:
        float or np.array

    """
    if base == -1:
        return np.log(ts).astype("float32")
    else:
        return (np.log(ts) / np.log(base)).astype("float32")


@njit("float32[:](float32[:], float32, int32)",cache=True, nogil=True, fastmath=True)
def apply_log_tail(ts, thresh, log_base):
    """Two-sided log.

    Args:
        ts (np.array)
        thresh (float)
            threshold to start log
        log_base (float)

    Returns:
        np.array

    """
    a = ts > thresh
    ts[a] = thresh + log(ts[a] + 1 - thresh, log_base)

    a = ts < -thresh

    ts[a] = -thresh - log(ts[a] * sign(ts[a]) + 1 - thresh, log_base)

    return ts
