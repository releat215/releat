"""Inference.

Functions necessary for real time inference
- calculating 1 record only
- appending the dataset to improve latency

"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl

from releat.data.cleaning import group_tick_data_by_time
from releat.data.cleaning import load_raw_tick_data
from releat.data.pipeline import make_feature
from releat.data.transformers import apply_transform
from releat.data.transformers import enrich_transform_config
from releat.data.utils import split_timeframe
from releat.utils.time import ceil_timestamp


def get_feature_time_range_in_seconds(trade_timeframe, feature_timeframe, feat_len):
    """Calcualte time range of feature in seconds.

    Calculate feature timeframe x feature length in seconds. For determining
    number of records to keep when initialising feature data

    Args:
        trade_timeframe (str):
            polars representation of time interval between observations /
            action time, i.e. 10s, 5m etc.
        feature_timeframe (str):
            polars representation of feature interval, i.e. 10s, 5m, etc
        feat_len (int):
            number of features in feat_group_ind

    Returns:
        recs (int):
            number of records to keep
        time_range (str):
            pandas string representation of the duration of the feat_len x
            feature_timeframe, i.e. 10S, 5T
        feat_time (str):
            pandas string representation of one feature timeframe, i.e. 10S, 5T

    """
    trade_time = int(trade_timeframe[:-1])
    assert trade_timeframe[-1] == "s"

    feat_time = int(feature_timeframe[:-1])
    unit = feature_timeframe[-1]

    if unit == "m":
        feat_time *= 60
    elif unit == "h":
        feat_time *= 60 * 60
    # one extra because of differencing features
    time_range = feat_len * feat_time + trade_time
    # records
    recs = int(time_range / trade_time)
    # add an extra one time range
    time_range += feat_time
    # using pandas notation
    time_range = str(int(time_range)) + "S"
    # one feature time
    feat_time = str(int(feat_time)) + "S"
    return recs, time_range, feat_time


def init_feature_dict(config, tick_data, now):
    """Initialise raw feature dict.

    Gets recs (from get_feature_time_range_in_seconds) number of values for each
    feature.

    Args:
        config (pydantic):
            output of config builder
        tick_data (dict[polars.Datafrae]):
            key is the symbol, value is the tick data
        now (pd.Timestamp):
            inference time as pandas timestamp

    Returns:
        dict[polars.Dataframe]:
            key1 = feat_group_ind, key2 = feat_ind, values for one feature

    """
    feature_data = {}
    dt = "inference"

    # get trade time from now
    trade_time = ceil_timestamp(
        deepcopy(now),
        config.raw_data.trade_timeframe,
        trade_time_offset=int(config.raw_data.trade_time_offset[0]),
    )
    trade_time -= pd.Timedelta(config.raw_data.trade_timeframe)

    for feat_group_ind in range(len(config.features)):
        feat_group = config.features[feat_group_ind]
        feature_data[feat_group_ind] = {}

        # get first and last time
        dt1 = ceil_timestamp(deepcopy(now), feat_group.timeframe, trade_time_offset=3)
        dt1 -= pd.Timedelta(feat_group.timeframe)

        recs, time_range, feat_time = get_feature_time_range_in_seconds(
            config.raw_data.trade_timeframe,
            feat_group.timeframe,
            feat_group.simple_features[0].output_shape[0],
        )
        dt0 = dt1 - pd.Timedelta(time_range)
        if dt0.dayofweek > 4:
            dt0 -= pd.Timedelta(hours=50)

        prev_symbol = ""

        for feat_ind in range(len(feat_group.simple_features)):
            fc = feat_group.simple_features[feat_ind]
            symbol = fc.symbol
            broker = fc.broker

            if symbol != prev_symbol:
                df = tick_data[f"{broker}_{symbol}"]

                # Get the last timestamp of the previous feature timeframe
                dt0 = df[df["time_msc"] < dt0]["time_msc"].iloc[-1] - pd.Timedelta(
                    feat_time,
                )
                dt0 = df[df["time_msc"] < dt0]["time_msc"].iloc[-1]

                df = df[(df["time_msc"] >= dt0) & (df["time_msc"] <= trade_time)]

                tick_df = load_raw_tick_data(config, broker, symbol, dt, df=df)
                tick_df = tick_df[1:]

                df_group = group_tick_data_by_time(
                    config,
                    feat_group_ind,
                    tick_df,
                    mode="train",
                )

                prev_symbol = symbol

            df = make_feature(
                df_group,
                config,
                feat_group_ind,
                feat_ind,
                fname=dt,
                mode="inference",
            )

            df = df.filter(pl.col("time_msc") <= trade_time)
            df = df.tail(recs)
            feature_data[feat_group_ind][feat_ind] = df

    latest_record = feature_data[0][0]["time_msc"][-1]

    # offset by an extra 1 s to avoid race conditions
    trade_timeframe = config.raw_data.trade_timeframe
    trade_timeframe, _ = split_timeframe(trade_timeframe)

    # This is necessary if the last 10s for a given timeframe is missing
    while trade_time > latest_record:
        # doesnt work for weekends, it tried to calculate every 10s in weekend
        new_trade_time = pd.to_datetime(latest_record) + pd.Timedelta(
            seconds=trade_timeframe,
        )
        day_of_week = new_trade_time.dayofweek
        while day_of_week >= 5:
            new_trade_time += pd.Timedelta(seconds=trade_timeframe)
            day_of_week = new_trade_time.dayofweek
        # update feature - note that new_trade_time must be a
        # multiple of the trade_timeframe
        feature_data = update_feature_dict(
            config,
            tick_data,
            feature_data,
            new_trade_time,
        )
        latest_record = feature_data[0][0]["time_msc"][-1]

    return feature_data


def update_feature_dict(config, tick_data, feature_data, trade_time):
    """Update feature dict.

    Args:
        config (pydantic):
            output from config builder
        tick_data (dict[polars.Dataframe])
            tick data
        feature_data (dict[polars.DataFrame])
            feature data
        trade_time (pd.Timedelta)
            trade time that is a multiple of the trade_timeframe

    Returns:
        dict[polars.DataFrame]
            feature_data updated with the next row

    """
    dt = "inference"

    for feat_group_ind in range(len(config.features)):
        feat_group = config.features[feat_group_ind]

        prev_symbol = ""

        for feat_ind in range(len(feat_group.simple_features)):
            fc = feat_group.simple_features[feat_ind]
            symbol = fc.symbol
            broker = fc.broker

            if symbol != prev_symbol:
                df = tick_data[f"{broker}_{symbol}"]
                dt0 = trade_time - pd.Timedelta(feat_group.timeframe.replace("m", "T"))
                dt0 = df[df["time_msc"] < dt0]["time_msc"].index[-1]

                df = df[dt0:]
                df = df[df["time_msc"] <= trade_time]

                tick_df = load_raw_tick_data(config, broker, symbol, dt, df=df)
                tick_df = tick_df[1:]

                df_group = group_tick_data_by_time(
                    config,
                    feat_group_ind,
                    tick_df,
                    mode="inference",
                )

                prev_symbol = symbol

            if len(tick_df) > 0:
                df = make_feature(
                    df_group,
                    config,
                    feat_group_ind,
                    feat_ind,
                    fname=dt,
                    mode="inference",
                )
                df = df.with_columns(
                    pl.lit(trade_time).dt.cast_time_unit("ns").alias("time_msc"),
                )

                assert len(df) == 1, print(len(df), df)
            else:
                df = feature_data[feat_group_ind][feat_ind][-1]
                df = df.with_columns(
                    pl.lit(trade_time).dt.cast_time_unit("ns").alias("time_msc"),
                )
                if fc.fillna != "foward":
                    for col in df.columns:
                        if col != "time_msc":
                            df = df.with_columns(
                                pl.lit(None).alias(col).cast(df[col].dtype),
                            )
            prev_value = feature_data[feat_group_ind][feat_ind]
            if df["time_msc"][-1] > prev_value["time_msc"][-1]:
                # this if statement is necessary because if the last 10s is missing by
                # higher timeframes already do have records
                feature_data[feat_group_ind][feat_ind] = pl.concat([prev_value, df])
                feature_data[feat_group_ind][feat_ind] = (
                    feature_data[feat_group_ind][feat_ind]
                    .fill_nan(None)
                    .fill_null(strategy=fc.fillna)
                )[1:]

    return feature_data


def get_obs_interval(trade_timeframe, feat_timeframe):
    """Get obs interval."""
    trade_val = int(trade_timeframe[:-1])
    # trade_unit = trade_timeframe[-1]

    val = int(feat_timeframe[:-1])
    unit = feat_timeframe[-1]
    mult = 1 / trade_val
    if unit == "m":
        mult = 60 / trade_val

    interval = int(val * mult)
    return interval


def make_feature_obs(config, feature_data):
    """Make feature obs."""
    obs = {}

    for feat_group_ind in range(len(config.features)):
        feat_group = config.features[feat_group_ind]
        feat_group_obs = []

        interval = get_obs_interval(config.raw_data.trade_timeframe, feat_group.timeframe)

        for feat_ind in range(len(feat_group.simple_features)):
            fc = feat_group.simple_features[feat_ind]

            feats = feature_data[feat_group_ind][feat_ind]
            cols = [x for x in feats.columns if x != "time_msc"]
            feats = feats[cols].to_numpy()[::interval].astype("float32")
            if fc.name == "differencing":
                # error is expected to be pip times larger than 1e-7
                feats = feats[:, 0]
                feats = feats - feats[-1]
                symbol = fc.symbol
                symbol_index = config.symbol_info_index[symbol]
                pip = config.symbol_info[symbol_index].pip
                feats = feats[:-1] / pip
                feats = feats.reshape((-1, 1))
                for tc in fc.transforms:
                    # tc = TransformerConfig(**tc)
                    feats = apply_transform(feats, tc)
            else:
                feats = feats[1:].astype("float32")
                for t_ind in range(len(fc.transforms)):
                    fc = enrich_transform_config(
                        config,
                        feat_group_ind,
                        feat_ind,
                        t_ind,
                        feat_shape=feats.shape,
                    )

                for tc in fc.transforms:
                    feats = apply_transform(feats, tc)

            feat_group_obs.append(feats)
        feat_group_obs = np.hstack(feat_group_obs)
        obs[str(feat_group_ind)] = feat_group_obs

    return obs
