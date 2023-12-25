"""Agent config.

TODO convert to yaml, how to pass parameters from
other files


"""
from __future__ import annotations

import os

from releat.utils.configs.constants import mt5_creds

agent_config = {
    "agent_version": "t00001",
    "platform": "mt5",
    "rl_algorithm": "impala",
    # tick data param
    "raw_data": {
        "tick_file_dates": ["2023-11-01"],
        "trade_timeframe": "10s",
        "tick_time_diff_clip_val": 60,
        "min_inf_time_s": "500ms",
        "max_inf_time_s": "2000ms",
        "trade_time_offset": "3s",
        "min_obs_val": -2.1,
        "max_obs_val": 2.1,
        "data_mode": "demo",
    },
    # gym env parameters
    "gym_env": {
        "hp_reload_t_s": 60,
        "skip_step": 3,
        "max_ep_step": 5000,
        "max_hold_t": 360,
        "min_hold_t": 7,
        "max_trades": 1000,
        "stop_val": -30.0,
        "min_ep_r": -1000.0,
        "max_ep_r": 1000.0,
        "commission": 0.4,
        "step_penalty": 0.0,
        "hold_penalty": 0,
        "eval_len": 10_000,
        "log_actions": False,
        "is_training": True,
        "train_iter": 1_000_000,
        "save_freq": 2,
        "rec_num": 10_000,
        "rec_growth": 0.05,
        "rec_warm_up": 20,
        "rec_ep_t": 5,
        "rec_reward_t": 1.0,
        "win_rate_t": 0.4,
        "pos_loss_t": -10.0,
        "drawdown_t": 30.0,
        "end_cum_r_t": -20.0,
        "min_cum_r_t": -30.0,
        "osample_p": 0.99,
        "osample_num": 10_000,
        "max_ep_repeats": 2,
        "mask_size_p": 0.5,
        "mask_dir_p": 0.05,
    },
    # aerospike configs (db)
    "aerospike": {
        "connection": {
            "hosts": [
                ("127.0.0.1", 3000),
            ],
            "policies": {
                "timeout": 1000,  # milliseconds
            },
        },
        "namespace": "prod",
    },
    "redis": {
        "host": "localhost",
        "port": 6369,
    },
    "mt5": mt5_creds["metaquotes"]["demo"],
    # trader configs
    "trader": {
        "trade_mode": "live",
        "deviation": 5,
        "lot": 0.01,
        "portfolio": [
            {
                "symbol": "EURUSD",
                "max_long": 2,
                "max_short": 2,
            },
        ],
    },
    # for each of the individual rllib components
    "rl_train": {
        "gamma": 0.99,
        "lr": 1e-5,
        # "lr_schedule": [
        #     [0, 0.0005],
        #     [100_000, 0.000000000001],
        # ],
        "train_batch_size": 1440,
        "model": {
            "custom_model_config": {
                # Depends on the architecture of agent_model.py
                "encoding_size": 16,
                "branch_layer_units": 16,
                "gr_units": 16,
                "gr_dropout": 1e-4,
                "actor_layer_depth": 2,
                "value_layer_depth": 2,
                "final_layer_units": 64,
            },
            "custom_model": "AgentModel",
        },
        # Impala Specific Configs
        "vtrace_drop_last_ts": False,
        "grad_clip": 40,
        "vf_loss_coeff": 0.005,
        "entropy_coeff": 0.01,
        "learner_queue_size": 16,
    },
    "rl_framework": {
        "framework": "tf",
    },
    "rl_rollouts": {
        # Specifying Rollout Workers
        "num_rollout_workers": 4,
        "num_envs_per_worker": 4,
        "rollout_fragment_length": 60,
        "batch_mode": "truncate_episodes",
        "ignore_worker_failures": False,
        "recreate_failed_workers": True,
    },
    "rl_explore": {
        "explore": True,
    },
    "rl_reporting": {
        "min_time_s_per_iteration": 30,
        "min_train_timesteps_per_iteration": 10_000,
    },
    "rl_debug": {
        "seed": 0,
    },
    "rl_resources": {
        # Resources
        "num_gpus": 1,
        "num_gpus_per_worker": 0,
        "num_cpus_per_worker": 1,
    },
    "rl_fault_tolerance": {
        "recreate_failed_workers": True,
    },
}

av = agent_config["agent_version"]
cwd = os.path.abspath(__file__)
assert av in cwd, "agent version: {av} does not match directory {cwd}"
