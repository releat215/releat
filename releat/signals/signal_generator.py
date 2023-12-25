"""Generate trade signal."""
from __future__ import annotations

import _pickle as cPickle
import os

import pandas as pd
import redis

from releat.signals.feature_builder import FeatureBuilder
from releat.signals.portfolio_manager import PortfolioManager
from releat.signals.rl_actor import RLActor
from releat.signals.tick_handler import TickHandler
from releat.utils.configs.config_builder import load_config
from releat.workflows.service_manager import start_mt5_api


class SignalGenerator:
    """Generates trade signal."""

    def __init__(self, agent_version):
        """Init.

        Args:
            self:
                None
            agent_version (str):
                i.e. 't00001'

        """
        self.agent_version = agent_version

        # load config
        self.config = load_config(agent_version, enrich_feat_spec=True, is_training=False)

        # start api for extracting data from mt5
        start_mt5_api("metaquotes", "general")

        # connect to redis - this is where the signals will be pushed
        self.redis = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            decode_responses=True,
        )

        # load rl agent
        self.rla = RLActor(agent_version, address="local")
        self.rla.reload_checkpoint()

        # tick handler extracts and aggregates data from mt5
        self.th = TickHandler(agent_version)

        # manages the gym and mt5 representation of active positions
        self.pm = PortfolioManager(agent_version)

        # transforms tick data into gym observations for rl agent
        self.fb = FeatureBuilder(agent_version)

    def initialise_data(self, now, clean_start=True):
        """Initialise data.

        Initialises tick and feature data. Where possible it loads a previously
        existing feature dataset. If testing different period, the init_feature_data.cpkl
        file needs to be deleted for a clean start.

        Args:
            now (str):
                datetime in the format "%Y-%m-%d %H:%M:%S.%f"
            clean_start (bool):
                if True, delete existing init_feature_data.cpkl


        """
        tmp_feat_file = f"{self.config.paths.trader_log_dir}/init_feature_data.cpkl"

        # delete temporary featre file for clean start
        if clean_start:
            if os.path.exists(tmp_feat_file):
                os.remove(tmp_feat_file)

        # Pretty hacky but should be done 3ish times
        # just in case the initialisation time >> trade time, i.e. if building initial
        # data takes 1 min but the trade time is 10s, then repeat to fill in the
        # missing features. Each run should be exponentially faster
        hour_delta = 72
        for _ in range(4):
            if os.path.exists(tmp_feat_file):
                with open(tmp_feat_file, "rb") as fobj:
                    self.feature_data = cPickle.load(fobj)

                # last timestep that we have feature data for
                last_feat_time = self.feature_data[0][0]["time_msc"][-1]

                # calculate how many ticks to pull from broker
                hour_delta = (
                    pd.to_datetime(now).tz_localize(None) - last_feat_time
                ).total_seconds() // 3600 + 1

                # at least pull 1 hr of data for initialisation
                hour_delta = max(hour_delta, 1)

            # load tick data
            self.th.init_tick_data(now, hour_delta)

            # load portfolio / positions
            self.pm.get_mt5_portfolio()
            self.pm.format_mt5_portfolio()
            self.fb.now = pd.to_datetime(now)
            self.fb.tick_data = self.th.tick_data
            self.fb.gym_portfolio = self.pm.gym_portfolio
            self.fb.gym_portfolio_hedge = self.pm.gym_portfolio_hedge

            # transform tick data to gym observations
            self.fb.init_feature_data()
            self.gym_obs_data = self.fb.make_gym_data()

    def update_data(self, now):
        """Update data.

        At each trade timestep, update the feature data.

        Args:
            now (str):
                datetime in the format "%Y-%m-%d %H:%M:%S.%f"

        """
        self.now = now

        # update tick data
        self.th.update_tick_data(now)

        # update feature data
        self.fb.update_feature_data()

        # update portfolio / positions
        self.pm.get_mt5_portfolio()
        self.pm.format_mt5_portfolio()
        self.fb.now = pd.to_datetime(now)
        self.fb.tick_data = self.th.tick_data
        self.fb.gym_portfolio = self.pm.gym_portfolio
        self.fb.gym_portfolio_hedge = self.pm.gym_portfolio_hedge

        # convert features into gym observation for inference
        self.gym_obs_data = self.fb.make_gym_data()

    def predict(self):
        """Predict Action.

        Using the gym observation, get the predicted action from rl agent.
        When different agents are running on different computes, push to a
        centralised redis, which can be picked up by traders.

        """
        pred = self.rla.predict(self.gym_obs_data)
        pred["time"] = self.now
        # push to redis as json
        self.redis.json().set(self.agent_version, "$", pred)
        return pred

    def check_tick_data(self):
        """Check tick data.

        Check that tick data is correctly appended. Possible some edge cases with MT5.
        For example when pulling data using time stamp range, micro seconds are ignored
        leading to possible tick duplication.

        """
        self.th.check_tick_data()

    def reload(self):
        """Reload checkpoint.

        Reload RL agent checkpoint to get the latest trained model.

        """
        self.rla.reload_checkpoint()
