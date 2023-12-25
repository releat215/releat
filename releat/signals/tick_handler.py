"""Gets and updates tick data.

Pulls data from apis/mt5.py

#TODO when we have more trading platform integrations,
abstract the api into a general tick data api

#TODO consider using multiprocessing or threading
and manager object for tick data to pull data
in parallel

Each rl_agent holds a copy of the tick data - should data be pushed into redis
and then each rl_agent pulls data from redis to do inference?

"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import timedelta

import pandas as pd
import pytz
import requests

from releat.data.extractor import download_tick_data
from releat.data.utils import update_tick_data
from releat.gym_env.action_processor import build_action_map
from releat.gym_env.action_processor import build_pos_arrs
from releat.utils.configs.config_builder import load_config
from releat.utils.configs.constants import mt5_api_port_map
from releat.utils.configs.constants import mt5_creds
from releat.utils.logging import get_logger
from releat.workflows.service_manager import start_mt5_api

logger = get_logger(__name__)


class TickHandler:
    """Tick data handler."""

    def __init__(self, agent_version):
        """Init.

        Args:
            agent_version (str):
                i.e. 't00001'

        """
        self.tick_data = None

        # load config
        self.config = load_config(agent_version, enrich_feat_spec=True, is_training=False)

        # action map - array where each row corresponds with an index in
        # the rl agents prediction. each column describes the action, i.e. long/short
        # open/close etc.
        self.action_map = build_action_map(self.config.trader)

        # initialise gym representation of positions
        self.gym_portfolio = build_pos_arrs(self.config.trader)
        self.gym_portfolio_hedge = build_pos_arrs(self.config.trader)

        # check available apis and start if necessary
        for si in self.config.symbol_info:
            port = mt5_api_port_map[si.broker][si.symbol]
            resp = requests.get(f"http://127.0.0.1:{port}/healthcheck")
            if resp.status_code != 200:
                logger.warning(f"Connection Error: http://127.0.0.1:{port}")
                start_mt5_api(si.broker, si.symbol)
                resp = requests.get(f"http://127.0.0.1:{port}/healthcheck")
                assert (
                    resp.stats_code == 200
                ), f"http://127.0.0.1:{port} failed to initialize"
            mt5_config = mt5_creds[si.broker][self.config.raw_data.data_mode]
            _ = requests.post(f"http://127.0.0.1:{port}/init", json=mt5_config)

        self.executor = ThreadPoolExecutor(len(self.config.symbol_info))

    def init_tick_data(self, dt1, hour_delta=72):
        """Initialise tick data.

        Download tick data from MT5
        #TODO switch dt1 to int? faster but harder to read
        #TODO dynamically generate how much data to hold in memory
        #TODO make all datetime references to UTC

        Args:
            dt1 (str):
                datetime in the format "%Y-%m-%d %H:%M:%S.%f" UTC
            hour_delta (int):
                number of hours of data to pull - generally only pull what is needed
                by the longest feature.

        """
        self.tick_data = {}

        # calculate start and end time
        dt1 = datetime.strptime(dt1, "%Y-%m-%d %H:%M:%S.%f")
        # TODO convert to local time
        dt1 = pytz.utc.localize(dt1)
        dt0 = dt1 - timedelta(hours=hour_delta)

        args = []
        for si in self.config.symbol_info:
            args = [si.broker, si.symbol]
            # TODO add in some broker timezone calculations
            args += [dt0, dt1]
            args += [self.config.raw_data.data_mode, False]

        # TODO check that this is correct for futures
        cols = ["ask", "bid", "flags", "last", "time_msc", "volume", "volume_real"]
        for result in self.executor.map(lambda p: download_tick_data(*args), args):
            tick_df = result["tick_df"][cols]
            broker = result["broker"]
            symbol = result["symbol"]
            self.tick_data[f"{broker}_{symbol}"] = tick_df

    def update_tick_data(self, dt1):
        """Update tick data.

        Download tick data delta between existing data and new datetime.

        Args:
            dt1 (str):
                new datetime for pulling data in the format "%Y-%m-%d %H:%M:%S.%f"

        """
        dt1 = datetime.strptime(dt1, "%Y-%m-%d %H:%M:%S.%f")
        new_data = {}
        dt0s = {}
        args = []
        for si in self.config.symbol_info:
            old_df = self.tick_data[f"{si.broker}_{si.symbol}"]
            dt0 = old_df["time_msc"].iloc[-1]
            dt0s[f"{si.broker}_{si.symbol}"] = dt0
            dt0 = dt0.replace(microsecond=0)
            dt0 = dt0.to_pydatetime()

            args = [si.broker, si.symbol]
            # TODO add in some broker timezone calculations
            args += [dt0, dt1]
            args += [self.config.raw_data.data_mode, False]

        # TODO check that this is correct for futures
        cols = ["ask", "bid", "flags", "last", "time_msc", "volume", "volume_real"]
        for result in self.executor.map(lambda p: download_tick_data(*args), args):
            broker = result["broker"]
            symbol = result["symbol"]
            tick_df = result["tick_df"][cols]
            dt0 = pd.to_datetime(dt0s[f"{broker}_{symbol}"]).replace(microsecond=0)
            tick_df = tick_df[tick_df["time_msc"] >= dt0]
            new_data[f"{broker}_{symbol}"] = tick_df

        # for symbol in self.symbols:
        #     old_df = self.tick_data[f"{symbol}"]
        #     dt0 = old_df["time_msc"].iloc[-1]
        #     dt0s[symbol] = dt0
        #     dt0 = dt0.replace(microsecond=0)

        #     payload = {
        #         "symbol": symbol,
        #         "dt0": dt0.strftime("%Y-%m-%d %H:%M:%S.%f"),
        #         "dt1": dt1,
        #     }
        #     df = requests.get(f"{self.data_api}/get_tick_data", json=payload).json()
        #     df = pd.DataFrame(df)[
        #         ["ask", "bid", "flags", "last", "time_msc", "volume", "volume_real"]
        #     ]
        #     df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms")

        #     df = df[df["time_msc"] >= pd.to_datetime(dt0)]
        #     new_data[symbol] = df
        self.tick_data = update_tick_data(self.tick_data, new_data)

        self.check_data = {}
        for broker_symbol in self.tick_data.keys():
            df = self.tick_data[broker_symbol]
            idx = df[df["time_msc"] == dt0s[broker_symbol]].index.tolist()[0]
            idx -= 10
            self.check_data[broker_symbol] = self.tick_data[broker_symbol].loc[idx:]

        # return self.tick_data

    def check_tick_data(self):
        """Check updated tick data.

        Check that tick data is appended correctly. This is to catch edge cases
        where data may have been pulled incorrectly. For examples microsecond are
        ignored or there might be increased latency, leading to duplicate
        or missing data.

        """
        # TODO check that this is correct for futures
        cols = ["ask", "bid", "flags", "last", "time_msc", "volume", "volume_real"]
        for si in self.config.symbol_info:
            # updated df
            udf = self.check_data[f"{si.broker}_{si.symbol}"]

            # get data from MT5 that spans the previous to new datetime range
            dt0 = pd.to_datetime(udf["time_msc"].iloc[0])
            dt1 = pd.to_datetime(udf["time_msc"].iloc[-1])

            result = download_tick_data(
                si.broker,
                si.symbol,
                dt0,
                dt1,
                self.config.raw_data.data_mode,
                False,
            )

            tick_df = result["tick_df"][cols]

            # ignore leading and trailing records de to microseconds
            tick_df = tick_df[tick_df["time_msc"] > dt0]
            tick_df = tick_df[tick_df["time_msc"] < dt1]
            tick_df.reset_index(drop=True, inplace=True)

            udf = udf[udf["time_msc"] > dt0]
            udf = udf[udf["time_msc"] < dt1]
            udf.reset_index(drop=True, inplace=True)

            # assert that the concatenated ticks are the same as the downloaded ticks
            if not pd.testing.assert_frame_equal(udf, tick_df):
                dt0 = dt0.strftime("%Y-%m-%d %H:%M:%S.%f")
                dt1 = dt1.strftime("%Y-%m-%d %H:%M:%S.%f")
                print(f"Error data append error: {dt0} - {dt1}")

        # for broker_symbol in self.check_data.keys():
        #     # updated df
        #     udf = self.check_data[broker_symbol]

        #     # get data from MT5 that spans the previous to new datetime range
        #     dt0 = udf["time_msc"].iloc[0]
        #     dt1 = udf["time_msc"].iloc[-1]
        #     payload = {
        #         "symbol": symbol,
        #         "dt0": dt0.strftime("%Y-%m-%d %H:%M:%S.%f"),
        #         "dt1": dt1.strftime("%Y-%m-%d %H:%M:%S.%f"),
        #     }
        #     df = requests.get(f"{self.data_api}/get_tick_data", json=payload).json()
        #     df = pd.DataFrame(df)[
        #         ["ask", "bid", "flags", "last", "time_msc", "volume", "volume_real"]
        #     ]
        #     df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms")
