"""Gym env."""
from __future__ import annotations

from copy import deepcopy
from time import time

import aerospike
import gymnasium as gym
import numpy as np
import pandas as pd

from releat.data.inference import get_obs_interval
from releat.data.simple.stats import randint
from releat.gym_env.action_processor import build_action_map
from releat.gym_env.action_processor import build_pos_arrs
from releat.gym_env.action_processor import exec_action
from releat.gym_env.action_processor import format_portfolio
from releat.gym_env.mask import assess_must_actions
from releat.gym_env.mask import make_mask
from releat.gym_env.metrics import TradingMetrics
from releat.gym_env.obs_processor import get_curr_price, portfolio_to_model_input
from releat.gym_env.obs_processor import get_obs
from releat.gym_env.obs_processor import get_raw_data


class FxEnv(gym.Env):
    """Forex gym env."""

    def __init__(self, config):
        """Init.

        Args:
            env_config (dict)

        Returns:
            None

        """
        super().__init__()

        self.config = deepcopy(config)
        # unpack env_config variables
        for k, v in config.items():
            setattr(self, k, v)

        self.client = aerospike.client(self.aerospike.connection).connect()

        self.meta_data_key = (
            self.aerospike.namespace,
            f"{self.aerospike.set_name}_hparams",
            "gym_env_configs",
        )

        # initialise action map
        self.action_map = build_action_map(self.trader)
        # initialise empty array for portfolio
        self.portfolio = build_pos_arrs(self.trader)

        for k, v in self.gym_env.dict().items():
            setattr(self, k, v)

        # initialise trade_journal
        if self.log_actions:
            self.trade_journal = np.zeros(
                (self.max_trades, self.portfolio.shape[1]),
                dtype="float64",
            )
        # timer for reloading max samples and new data
        self.reload_metadata_t0 = 0
        self.start_ind = None
        self.trading_metrics = TradingMetrics()

        raw_data_shape = {}
        obs_interval = {}
        lens = []
        trade_timeframe = self.raw_data.trade_timeframe
        for i in range(len(self.features)):
            feat_group = self.features[i]
            feat_timeframe = feat_group.timeframe
            feat_len = feat_group.simple_features[0].output_shape[0]
            interval = get_obs_interval(trade_timeframe, feat_timeframe)
            obs_interval[str(i)] = interval
            raw_data_shape[i] = feat_len
            lens.append(feat_len)
        raw_data_shape["max"] = max(lens)
        self.raw_data_shape = raw_data_shape
        self.obs_interval = obs_interval

    def initialize(self):
        """Initialize env.

        Gets starting parameters for counters, etc.

        """
        _ = np.random.seed()

        # check if we need to update training metadata
        t = time()
        if t - self.reload_metadata_t0 > self.hp_reload_t_s:
            self.reload_metadata_t0 = t
            _, _, bins = self.client.get(self.meta_data_key)
            for k, v in bins.items():
                if k not in ["is_training", "log_actions"]:
                    setattr(self, k, v)

            self.trading_metrics.set_max_repeat_num(self.max_ep_repeats)

            if self.is_training:
                # exclude eval data from training data
                self.max_data_ind -= self.eval_len
            else:
                self.skip_step = 1
                self.max_ep_step = 1_000_000_000

        if self.log_actions:
            self.action_log = []
        # initialise empty array for portfolio
        self.portfolio[:, 5:] = self.portfolio[:, 5:] * 0.0

        # Reset the state of the environment to an initial state
        self.rewards = 0
        self.ep_time = 0

        # initialise trade_journal
        if self.log_actions:
            self.trade_journal *= 0
        self.trade_journal_ind = 0

        # TODO make parametric / different view of position value
        self.raw_pos_vals = [[0.0, 0.0] * len(self.symbol_info)] * 1

    def reset(self, *, seed=None, options=None):
        """Reset env."""
        _ = self.initialize()
        _ = self.trading_metrics.decide_repeat(
            self.win_rate_t,
            self.pos_loss_t,
            self.drawdown_t,
            self.end_cum_r_t,
            self.min_cum_r_t,
        )
        # randomly initialise first global map id if train mode
        if self.is_training:
            if self.mask_size_p > np.random.rand():
                # number of positions
                # TODO - convert to random number (1,2,3) rather than True False
                self.mask_pos_size = True
            else:
                self.mask_pos_size = False

            rand = np.random.rand()
            if self.mask_dir_p > rand:
                # Ep can only long or only short
                self.mask_pos_dir = -1
            elif 1 - self.mask_dir_p < rand:
                self.mask_pos_dir = 1
            else:
                self.mask_pos_dir = 0

            if (not self.trading_metrics.repeat_ep) | (self.start_ind is None):
                if np.random.rand() > self.osample_p:
                    low = max(self.max_data_ind - self.osample_num, 50000)
                else:
                    low = max(
                        self.max_data_ind - self.max_samples - int(self.max_ep_step / 2),
                        50000,
                    )
                high = self.max_data_ind - int(self.max_ep_step / 2)
                self.start_ind = randint(low, high)
            self.data_ind = randint(self.start_ind - 100, self.start_ind + 100)
        else:
            if seed is None:
                # if debug or test mode, start from eval time
                self.data_ind = self.start_ind = (
                    self.max_data_ind - self.eval_len + randint(-20, 20)
                )
            else:
                self.data_ind = self.start_ind = seed
            self.mask_pos_size = False
            self.mask_pos_dir = 0

        self.trading_metrics.reset_metrics(self.start_ind)


        return self._next_observation(), {}

    def _next_observation(self):
        """Next observation."""
        
        self.data = get_raw_data(
            self.config,
            self.client,
            self.raw_data_shape,
            self.obs_interval,
            self.data_ind,
        )

        # potentially a bit dodgy because of leaking future data into current timestep
        price = self.data["trade_price"]
        self.curr_price = get_curr_price(self.symbol_info, price)
        self.time_int = self.data["date"]
        self.portfolio, _, _ = exec_action(
            self.action_map,
            self.portfolio,
            0,
            self.curr_price,
            self.time_int,
            self.commission,
        )
        
        pos_val = portfolio_to_model_input(self.portfolio).tolist()
        self.raw_pos_vals.append(pos_val)
        self.raw_pos_vals = self.raw_pos_vals[-1:]
        
        obs = get_obs(self.config, self.obs_interval, deepcopy(self.data))        

        must_hold, must_close = assess_must_actions(
            self.portfolio,
            self.ep_time,
            self.max_ep_step,
            self.min_hold_t,
            self.max_hold_t,
            self.data_ind,
            self.max_data_ind,
        )

        obs["mask"] = make_mask(
            self.action_map,
            self.portfolio,
            self.stop_val,
            must_hold,
            must_close,
            self.mask_pos_size,
            self.mask_pos_dir,
        )

        obs["pos_val"] = np.array(self.raw_pos_vals, dtype="float32")
        return obs

    def step(self, action):
        """Execute action.

        Args:
            action (int)
                action taken by agent

        Returns:
            None

        """
        # take action
        self.portfolio, reward, trade_journal_entry = exec_action(
            self.action_map,
            self.portfolio,
            action,
            self.curr_price,
            self.time_int,
            self.commission,
        )

        has_closed = np.argwhere(self.action_map[:, 2] == -1)

        tmp = np.array(self.raw_pos_vals)
        for i in range(len(self.portfolio)):
            if (action in has_closed) & (self.portfolio[i, 5] == 0):
                tmp[:, i] *= 0
                tmp[:, i + 1] *= 0
        self.raw_pos_vals = list(tmp)

        if trade_journal_entry is not None:
            if self.log_actions:
                self.trade_journal[self.trade_journal_ind] = trade_journal_entry.copy()
            self.trade_journal_ind += 1

        _ = self.trading_metrics.update_metrics(reward, self.rewards + reward)

        # TODO abstract reward function
        # step penalty
        reward -= self.step_penalty
        reward = np.clip(reward / 30, -10.0, 10.0)

        done = False
        if self.is_training:
            # if number of trades is exceeded
            done = self.trade_journal_ind >= self.max_trades
            # if episode time is exceeded
            done = done | (self.ep_time >= self.max_ep_step)
            # if losses are exceeded
            done = done | (self.rewards < self.min_ep_r)
            # if wins are exceeded
            done = done | (self.rewards > self.max_ep_r)

        # if end of available data
        done = done | (self.data_ind >= self.max_data_ind - 10)
        # no open positions (i.e. sum of pos_size = 0)
        done = done & (self.portfolio[:, 5].sum() == 0)

        if self.log_actions:
            # save rewards
            log = [
                pd.Series(
                    [self.data_ind, self.data["date"], action, reward, self.rewards],
                    index=["data_ind", "action_time", "action", "reward", "cum_rewards"],
                ),
            ]
            df = format_portfolio(self.symbol_info, self.portfolio)
            for i in df.index:
                tmp = df.loc[i].copy()
                if tmp["curr_price"] == 0:
                    tmp["curr_price"] = np.mean(self.curr_price[i])
                tmp.index = [f"{i}_{c}" for c in tmp.index]
                log.append(tmp)

            log = pd.concat(log, axis=0)
            self.action_log.append(log)

        # update counters and state
        self.ep_time += 1
        self.rewards += reward
        self.data_ind += self.skip_step
        
        obs = self._next_observation()

        return obs, reward, done, False, {}

    def get_results(self):
        """Get result."""
        cp = format_portfolio(self.symbol_info, self.trade_journal).head(
            self.trade_journal_ind,
        )
        cp["curr_val"] = cp["curr_val"] - 0.4

        res = pd.concat(self.action_log, axis=1).T
        res["action_time"] = pd.to_datetime(res["action_time"], unit="s").dt.strftime(
            "%Y-%m-%d %H:%M:%S",
        )
        return res, cp
