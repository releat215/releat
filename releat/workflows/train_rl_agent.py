"""Train rl agent.

Process for training agent

"""
from __future__ import annotations

import os
from glob import glob
from time import time

import aerospike
import ray
from ray.rllib.algorithms.impala import ImpalaConfig as RLAlgorithmConfig
from ray.rllib.models import ModelCatalog

from releat.gym_env.gym_env import FxEnv


def train_rl_agent(config, AgentModel):
    """Train rl agent."""
    client = aerospike.client(config.aerospike.connection).connect()

    key = (
        config.aerospike.namespace,
        config.aerospike.set_name + "_hparams",
        "gym_env_configs",
    )

    _, _, bins = client.get(key)
    max_samples = bins["max_samples"]

    ray.init(address="auto")

    logdir = config.paths.algo_dir
    # ckpt_offset = max([int(x.split("/")[-1].strip()) for x in glob(f"{logdir}/0*")])
    _ = os.makedirs(logdir, exist_ok=True)

    ModelCatalog.register_custom_model("AgentModel", AgentModel)

    env_config = {
        **config.rl_env,
        "env_config": dict(config),
    }

    trainer = (
        RLAlgorithmConfig()
        .training(**config.rl_train)
        .environment(env=FxEnv, **env_config)
        .framework(**config.rl_framework)
        .rollouts(**config.rl_rollouts)
        .exploration(**config.rl_explore)
        .reporting(**config.rl_reporting)
        .debugging(
            logger_config={"type": "ray.tune.logger.TBXLogger", "logdir": logdir},
            **config.rl_debug,
        )
        .resources(**config.rl_resources)
        .build()
    )

    trainer.get_policy().model.base_model.summary()

    try:
        files = sorted(glob(f"{logdir}/checkpoint*"))
        f = sorted(glob(f"{files[-1]}/*"))
        trainer.restore(f[0])
    except Exception as e:
        print(str(repr(e)))

    t0 = time()

    counter = 0

    _, _, bins = client.get(key)

    for i in range(bins["train_iter"]):
        results = trainer.train()
        counter += 1

        print(
            f"{results['timesteps_total']/1_000_000:.1f}".rjust(7),
            f"| {max_samples/1_000_000:.2f}".rjust(6),
            f"| reward: {results['episode_reward_mean']:.1f}".rjust(13),
            f"| len: {results['episode_len_mean']:.0f}".rjust(9),
            f"| eps: {results['episodes_this_iter']}".rjust(7),
            end="",
        )

        _, _, bins = client.get(key)
        max_samples = bins["max_samples"]

        if (
            (i > bins["rec_warm_up"])
            & (counter > bins["rec_ep_t"])
            & (results["episode_reward_mean"] > bins["rec_reward_t"])
        ):
            add_amt = min(max_samples * bins["rec_growth"], 100_000)
            max_samples = min(max_samples + add_amt, bins["max_data_ind"] - 5)
            bins["max_samples"] = int(max_samples)
            _ = client.put(key, bins)
            counter = 0

        t1 = time()
        tt = t1 - t0
        ckpt_print_str = "time: " + f"{tt:.0f}s".ljust(5)

        t0 = t1

        if (i + 1) % bins["save_freq"] == 0:
            checkpoint = trainer.save(logdir)
            checkpoint_str = checkpoint.split("/")[-1].split("_")[-1]
            ckpt_print_str += f"  ckpt: {checkpoint_str}"

        print(" | " + ckpt_print_str)
