import argparse
import gym
import gym
import ray
from ray import tune
from ray.rllib.policy.policy import PolicySpec
from env_H2SCM import HydroRefuelSys
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import  HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.rllib.agents.ppo import ppo

trial = 1

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    # multi-agent args
    parser.add_argument("--num-HRS", type=int, default=2)

    # general args
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "tfe", "torch"],
        default="tf",
        help="The DL framework specifier.",
    )
    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(ignore_reinit_error = True)

    HDS_config = {
        "lr": tune.uniform(1e-7, 1e-4),
        "train_batch_size" : tune.uniform(300,3000)
    }
    HRS0_config = {
        "lr": tune.uniform(1e-7, 1e-4),
        "train_batch_size": tune.uniform(300,3000)
    }
    HRS1_config = {
        "lr": tune.uniform(1e-7, 1e-4),
        "train_batch_size": tune.uniform(300, 3000)
    }

    def policy_mapping_fn(agent_id, episdoe, worker, **kwargs):
        if agent_id.startswith("HRS"):
            policy_id = agent_id[:4]
            return policy_id
        else:
            return agent_id

    algo = HyperOptSearch()
    scheduler = AsyncHyperBandScheduler()

    tune.run(
        "PPO",
        local_dir=f'~/MARL_results/TUNE/trial{trial}',
        # stop=stop,
        config={
            "env": HydroRefuelSys,
            "env_config": {
              "num_HRS": args.num_HRS,
                # "G_pie_max": [4.08, 11.34],
                # "B_pie_max": [3.17, 10.43],
                # "X_pie_max": [2.74, 10.0]
            },
            "disable_env_checking": True,
            "num_workers": 1,
            "multiagent": {
                "policies": {
                    "HDS": PolicySpec(observation_space=gym.spaces.Box(0, 100000000, shape=(4,3)), 
                                      action_space=gym.spaces.Box(0, 1, shape=(9,)), config=HDS_config),
                    "HRS0": PolicySpec(observation_space=gym.spaces.Box(0, 100000000, shape=(8,)), 
                                       action_space=gym.spaces.Box(0, 1, shape=(5,)), config=HRS0_config),
                    "HRS1": PolicySpec(observation_space=gym.spaces.Box(0, 10000, shape=(8,)), 
                                       action_space=gym.spaces.Box(0, 1, shape=(3,)), config=HRS1_config)
                },
                "policy_mapping_fn": policy_mapping_fn,
            },
            "framework": args.framework,
        },
        metric = "episode_reward_mean",
        mode = "max",
        search_alg = algo,
        scheduler = scheduler,
        num_samples = 1500
    )
