"""Transformers.

Transforms features by:
- scaling
- clipping
- normalization

"""
from __future__ import annotations

import _pickle as cPickle
import os
import pickle
from copy import deepcopy
from glob import glob

import numpy as np
import pandas as pd
import polars as pl
from numba import njit
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm

from releat.data.utils import get_feature_dir
from releat.data.utils import split_timeframe
from releat.utils.logging import get_logger

logger = get_logger(__name__)


@njit(cache=True, nogil=True, fastmath=True)
def find_clip_values(arr, lower_lim, upper_lim, method):
    """Find clip values.

    # TODO only tested for 2D data

    Args:
        arr (np.array)
        lower_lim (np.float)
        upper_lim (np.float)
        method (str)
            either 'value' or 'percentile'

    Returns:
        dict(np.array)
            min and max limits

    """
    out_len = arr.shape[-1]
    clip_min = np.ones(out_len, dtype="float32")
    clip_max = np.ones(out_len, dtype="float32")

    if method == "value":
        clip_min *= lower_lim
        clip_max *= upper_lim
    elif method == "percentile":
        for i in range(out_len):
            clip_min[i] = np.percentile(arr[:, i], lower_lim)
            clip_max[i] = np.percentile(arr[:, i], upper_lim)
    clip_vals = {"clip_min": clip_min, "clip_max": clip_max}

    return clip_vals


@njit(cache=True, nogil=True, fastmath=True)
def clip_by_value(arr, clip_min, clip_max, scale_factor):
    """Clip by value.

    # each must be the same - arr must be 2d (n,m)
    # clip vals must be of shape (n,m)

    Args:
        arr (np.array)
            must be 2d array
        clip_min (np.array)
            min value of array - shape must be same as arr
        clip_max (np.array)
            max value of array - shape must be same as arr
        scale_factor (float)
            float to scale up or down the array after clipping

    Returns:
        np.array:
            scaled array - same shape as arr

    """
    for i in range(arr.shape[1]):
        a = arr[:, i] < clip_min[:, i]
        b = arr[:, i] > clip_max[:, i]
        if a.sum() > 0:
            arr[a, i] = clip_min[a, i]
        if b.sum() > 0:
            arr[b, i] = clip_max[b, i]
    arr *= scale_factor
    return arr


def apply_clip(arr, tc):
    """Apply clip.

    Non-jit wrapper function for clipping

    Args:
        arr (np.array)
        tc (dict)

    Returns:
        np.array
            clipped array

    """
    arr = clip_by_value(arr, tc.clip_min, tc.clip_max, tc.scale_factor)
    return arr


def fit_scaler(arr, method):
    """Fit scaler.

    # arr must be cols = features and index = samples

    Args:
        arr (np.array)
        method (str)
            'PowerTransformer' is the only option for now, but planning for other
            transformers

    Returns:
        arr (np.array)
            scaling parameters

    """
    if method == "PowerTransformer":
        s = PowerTransformer(method="yeo-johnson", standardize=True)
        _ = s.fit(arr)
        out = {
            "mean": s._scaler.mean_.astype("float32"),
            "std": s._scaler.scale_.astype("float32"),
            "lambda": s.lambdas_.astype("float32"),
        }
        return out


@njit(cache=True, nogil=True, fastmath=True)
def yeo_johnson_transform_vec(lmbda, mean, std, vec):
    """Yeo-johnson transform for vector.

    Return transformed input vec following Yeo-Johnson transform with
    parameter lambda.Element wise, so each input must have same length
    Does one column only

    Args:
        lmnda (np.array)
        mean (np.array)
        std (np.array)
        vec (np.array)

    Returns:
        np.array

    """
    out = np.zeros_like(vec, dtype="float32")
    space = np.ones(vec.shape, dtype="float32") * np.float32(np.spacing(1.0))

    pos = vec >= 0  # binary mask

    # when vec > 0
    a = np.abs(lmbda) < space
    b = pos & a
    out[b] = np.log1p(vec[b])

    # lmbda != 0
    b = pos & ~a
    out[b] = ((vec[b] + 1) ** lmbda[b] - 1) / lmbda[b]

    # when vec < 0
    a = np.abs(lmbda - 2) > space
    b = ~pos & a
    out[b] = -((-vec[b] + 1) ** (2 - lmbda[b]) - 1) / (2 - lmbda[b])

    b = ~pos & ~a
    out[b] = -np.log1p(-vec[b])

    # standardise
    out -= mean
    out /= std

    return out


@njit(cache=True,nogil=True, fastmath=True)
def yeo_johnson_transform(lmbda, mean, std, arr):
    """Yeo-johnson transform for array.

    calls the vector transform for each column

    Args:
        lmnda (np.array)
        mean (np.array)
        std (np.array)
        vec (np.array)

    Returns:
        np.array

    """
    # all must be the same shape (n,m)
    for i in range(arr.shape[1]):
        arr[:, i] = yeo_johnson_transform_vec(
            lmbda[:, i],
            mean[:, i],
            std[:, i],
            arr[:, i],
        )
    return arr


@njit(cache=True,nogil=True, fastmath=True)
def linear_scaling(arr):
    """Linear scaling.

    Todo make the scaling more parametric

    Args:
        arr (np.array)

    Returns:
        np.array

    """
    for i in range(arr.shape[1]):
        tmp = arr[:, i]
        lim1 = 2
        lim2 = 3
        a = tmp > lim1
        tmp[a] = lim1 + (tmp[a] - lim1) / 2
        a = tmp > lim2
        tmp[a] = lim2 + (tmp[a] - lim2) / 2

        a = tmp < -lim1
        tmp[a] = -lim1 + (tmp[a] + lim1) / 2
        a = tmp < -lim2
        tmp[a] = -lim2 + (tmp[a] + lim2) / 2

        arr[:, i] = tmp

    return arr


def apply_scaling(arr, tc):
    """Apply scaling.

    Wrapper function for scaling

    Args:
        arr (np.array)
        tc (dict)

    Returns:
        scaled array

    """
    if tc.method == "PowerTransformer":
        return yeo_johnson_transform(tc.lam, tc.mean, tc.std, arr)
    elif tc.method == "PiecewiseLinear":
        return linear_scaling(arr)


def apply_transform(arr, tc):
    """Apply transform.

    Args:
        arr (np.array)
        tc (dict)

    Returns:
        np.array
            transformed array

    """
    name = tc.name
    transforms = {
        "clip": apply_clip,
        "scale": apply_scaling,
    }
    func = transforms[name]
    return func(arr, tc)


def get_one_transform_param(config, feat_group_ind, feat_ind, t_ind, feats):
    """Get one transform param.

    Calculates the transform parameters of one config

    data should always be (X,1)

    Args:
        folder (str)
        fc (dict)
        feat_ind (int)
        t_ind (int)
        feats (np.array)
            note we overwrite feats each time we call this function

    Returns:
        np.array
            transformed feats

    """
    feat_group = config.features[feat_group_ind]
    fc = feat_group.simple_features[feat_ind]
    tc = fc.transforms[t_ind]

    folder = get_feature_dir(config, feat_group_ind, feat_ind)
    folder = f"{folder}/transforms"
    _ = os.makedirs(folder, exist_ok=True)
    transform_f = f"{folder}/{t_ind}_{tc.name}_{tc.method}.cpkl"

    ones = np.ones(feats.shape)

    if tc.name == "clip":
        clip_vals = find_clip_values(
            feats,
            tc.lower_lim,
            tc.upper_lim,
            tc.method,
        )

        with open(transform_f, "wb") as fobj:
            cPickle.dump(dict(clip_vals), fobj)

        tc.clip_min = clip_vals["clip_min"] * ones
        tc.clip_max = clip_vals["clip_max"] * ones

    if tc.name == "scale":
        if tc.method == "PowerTransformer":
            scaler = fit_scaler(feats, tc.method)

            with open(transform_f, "wb") as fobj:
                cPickle.dump(scaler, fobj)

            tc.lam = scaler["lambda"] * ones
            tc.mean = scaler["mean"] * ones
            tc.std = scaler["std"] * ones

    feats = apply_transform(feats, tc)

    return feats


def get_transform_params(config, feat_group_ind, feat_ind, feats):
    """Get transform params.

    Calculates all the transform parameters for a config and returns the transformed
    feat (so that it can be an input into the next transformation)

    Args:
        folder (str)
        fc (dict)
        feat_ind (int)
        feats (np.array)
        ft (np.array)
            data of full timeframe (not partial)

    Returns:
        np.array
            transformed feats

    """
    feat_group = config.features[feat_group_ind]
    fc = feat_group.simple_features[feat_ind]

    # folder = get_feature_dir(config, feat_group_ind, feat_ind)
    # plot_f = f"{folder}/transforms/hist_and_qq_plots"

    for t_ind in range(len(fc.transforms)):
        # print(f'transform {t_ind}')
        feats = get_one_transform_param(config, feat_group_ind, feat_ind, t_ind, feats)
        # print('transform done')

    assert not np.isnan(feats).all(), "some nans in feat"
    # title = f"{feat_ind}_{fc.symbol}_{fc.timeframe}_{fc.name}"
    # _ = plot_hist_and_qq(feats, title, plot_f)

    return feats


def get_transform_params_for_one_feature_group(config, feat_group_ind):
    """Get transform param for one group.

    For all the features in one feature group, calculate the weights / values for
    the transformations

    Args:
        config (pydantic.BaseModel):
            as defined in 'agent_config.py'
        feat_group_ind (int):
            index of the feature group

    Returns:
        None

    """
    feat_group = config.features[feat_group_ind]
    for feat_ind in range(len(feat_group.simple_features)):
        logger.info(f"Scaling feature {feat_group_ind}-{feat_ind}")
        feature_dir = get_feature_dir(config, feat_group_ind, feat_ind)
        files = list(sorted(glob(f"{feature_dir}/raw_data/*")))
        dfs = None
        for f in tqdm(files):
            df_raw = pl.read_parquet(f)
            if feat_group.simple_features[feat_ind].name == "differencing":
                df = df_raw.with_columns(
                    pl.Series(list(range(len(df_raw)))).alias("group_ind"),
                )

                feat_group = config.features[feat_group_ind]
                fc = feat_group.simple_features[feat_ind]
                symbol_index = config.symbol_info_index[fc.symbol]
                pip = config.symbol_info[symbol_index].pip

                obs_len = fc.output_shape[0]
                num, unit = split_timeframe(fc.timeframe)
                # TODO clean up this logic
                # if the unit is seconds divide by 10 (because 10s intervals) otherwise
                # divide by 1 - note its 0.1 because we have 10s intervals
                mult = 6 if unit == "m" else 0.1
                obs_timeframe = str(int(obs_len * num * mult)) + "i"
                # for now differencing is only for the dataframes with one feature
                col = "feat"

                if unit == "s":
                    take_every_num = int(num / 10)
                elif unit == "m":
                    # TODO instead of 6 it should be 60seconds / trade timeframe
                    take_every_num = int(num * 60 / 10)

                df_obs_group = df.set_sorted("group_ind").groupby_dynamic(
                    "group_ind",
                    every="1i",
                    period=obs_timeframe,
                    closed="both",
                )

                df = (
                    df_obs_group.agg(
                        [
                            ((pl.col(col) - pl.col(col).last()) / pip).alias(col),
                        ],
                    )
                    .with_columns(
                        pl.col(col).apply(
                            lambda x: x.take_every(take_every_num).head(obs_len),
                        ),
                    )
                    .filter(pl.col("group_ind") >= 0)
                    .with_columns(pl.lit(df_raw["time_msc"]).alias("time_msc"))
                )
                df = df.head(len(df) - int(obs_len * num * mult))

            else:
                df = df_raw

            if dfs is None:
                dfs = df
            else:
                last_dt = dfs["time_msc"][-1]
                assert last_dt > df["time_msc"][0]
                df = df.filter(pl.col("time_msc") > last_dt)
                dfs = pl.concat([dfs, df], how="vertical")

        if feat_group.simple_features[feat_ind].name == "differencing":
            feats = dfs.select(pl.col("feat").list.to_struct()).unnest("feat").to_numpy()
        else:
            cols = [x for x in dfs.columns if x != "time_msc"]
            feats = dfs.select(cols).to_numpy()

        _ = get_transform_params(config, feat_group_ind, feat_ind, feats)



def get_transform_params_for_all_features(config):
    """Get transforms for groups."""
    for feat_group_ind in range(len(config.features)):
        _ = get_transform_params_for_one_feature_group(config, feat_group_ind)


def enrich_transform_config(config, feat_group_ind, feat_ind, t_ind, feat_shape=None):
    """Enrich one transform config.

    Loads transform configs into dict
    # when doing batch transforms, we need the feature size
    # when in real time, we can pass through just the dimension of 1 batch
    # enrich transform

    Args:
        folder (str)
        fc (dict)
        feat_ind (int)
        t_ind (int)
        feat_size (tuple())

    Returns:
        dict
            one tc with scaled params

    """
    feat_group = config.features[feat_group_ind]
    fc = feat_group.simple_features[feat_ind]

    folder = get_feature_dir(config, feat_group_ind, feat_ind)

    tc = fc.transforms[t_ind]
    transform_f = f"{folder}/transforms/{t_ind}_{tc.name}_{tc.method}.cpkl"

    if not tc.method == "PiecewiseLinear":
        with open(transform_f, "rb") as fobj:
            scaler = cPickle.load(fobj)

        if feat_shape is None:
            feat_shape = np.ones(fc.output_shape)
        else:
            feat_shape = np.ones(feat_shape)

        match tc.name:
            case "clip":
                if fc.name == "differencing":
                    tc.clip_min = scaler["clip_min"].reshape(fc.output_shape)
                    tc.clip_max = scaler["clip_max"].reshape(fc.output_shape)

                else:
                    tc.clip_min = scaler["clip_min"] * feat_shape
                    tc.clip_max = scaler["clip_max"] * feat_shape
            case "scale":
                if fc.name == "differencing":
                    tc.lam = scaler["lambda"].reshape(fc.output_shape)
                    tc.mean = scaler["mean"].reshape(fc.output_shape)
                    tc.std = scaler["std"].reshape(fc.output_shape)
                else:
                    tc.lam = scaler["lambda"] * feat_shape
                    tc.mean = scaler["mean"] * feat_shape
                    tc.std = scaler["std"] * feat_shape

        fc.transforms[t_ind] = tc
    return fc


def enrich_all_feature_configs(config):
    """Enich all feature configs.

    Loads all transform and feature transform configs into the dict

    Args:
        data_config (dict)
        feature_configs (dict)

    Returns:
        dict
            feature_configs with all arrays necessary for scaling and transforming

    """
    for feat_group_ind in range(len(config.features)):
        feat_group = config.features[feat_group_ind]
        for feat_ind in range(len(feat_group.simple_features)):
            fc = feat_group.simple_features[feat_ind]

            for t_ind in range(len(fc.transforms)):
                fc = enrich_transform_config(config, feat_group_ind, feat_ind, t_ind)

            config.features[feat_group_ind].simple_features[feat_ind] = deepcopy(fc)

    # TODO possible remove this
    # config.rl_env["gym_env"]["feature"] = config.features
    return config


def scale_feature(config, feat_group_ind, feat_ind, dt):
    """Scale feature.

    Make sure enrich config is called before this function. Apply the transforms
    to the raw features

    Args:
        config (pydantic.BaseModel):
            as defined in 'agent_config.py'
        feat_group_ind (int):
            index of feature group
        feat_ind (int):
            index of feature within feature group
        dt (pd.DateTime):
            datetime of the file that is loaded, usually in the format of '%Y-%m-01'
            because train data is built using one months worth of ticks

    """
    feature_dir = get_feature_dir(config, feat_group_ind, feat_ind)
    df_raw = pd.read_parquet(f"{feature_dir}/raw_data/{dt}.parquet", engine="pyarrow")

    fc = config.features[feat_group_ind].simple_features[feat_ind]
    scaled_obs_dir = get_feature_dir(config, feat_group_ind, feat_ind)
    scaled_obs_dir = f"{scaled_obs_dir}/scaled_data"
    os.makedirs(scaled_obs_dir, exist_ok=True)

    feat = df_raw.drop("time_msc", axis=1).to_numpy()

    def flatten_polar_array(arr):
        return np.stack(arr).T

    fc = config.features[feat_group_ind].simple_features[feat_ind]

    feat = np.apply_along_axis(flatten_polar_array, 0, feat).astype("float32")
    feat = np.moveaxis(feat, 0, 1)

    for i in range(len(feat)):
        for tc in fc.transforms:
            feat[i] = apply_transform(feat[i], tc)

    feat = pd.Series(list(feat), name=str(feat_ind)).to_frame()
    feat.index = df_raw["time_msc"].tolist()

    feat[str(feat_ind)] = feat[str(feat_ind)].apply(pickle.dumps)

    feat.to_parquet(f"{scaled_obs_dir}/{dt}.parquet")
