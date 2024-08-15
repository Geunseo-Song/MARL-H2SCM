## tuning two policy at same time for env2
import argparse
import gym
import gym
import ray
from ray import tune
from ray.rllib.policy.policy import PolicySpec
from env import HydroRefuelSys
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import  HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.rllib.agents.ppo import ppo

trial = 7

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    # multi-agent args
    parser.add_argument("--num-HRS", type=int, default=2)

    # general args
    # parser.add_argument(
    #     "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    # )
    # parser.add_argument("--num-cpus", type=int, default=2)
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "tfe", "torch"],
        default="tf",
        help="The DL framework specifier.",
    )
    # parser.add_argument("--eager-tracing", action="store_true")
    # parser.add_argument(
    #     "--stop-iters", type=int, default=1000000, help="Number of iterations to train."
    # )
    # parser.add_argument(
    #     "--stop-timesteps",
    #     type=int,
    #     default=50,
    #     help="Number of timesteps to train.",
    # )
    # parser.add_argument(
    #     "--stop-reward",
    #     type=float,
    #     default=80.0,
    #     help="Reward at which we stop training.",
    # )
    # parser.add_argument(
    #     "--local-mode",
    #     action="store_true",
    #     help="Init Ray in local mode for easier debugging.",
    # )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(ignore_reinit_error = True)
    # ray.init(num_cpus=args.num_cpus or None)

    # stop = {
    #     "training_iteration": args.stop_iters,
    #     # "timesteps_total": args.stop_timesteps,
    #     # "episode_reward_mean": args.stop_reward,
    # }

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
    # policies = {f"HRS{i}": PolicySpec(observation_space=gym.spaces.Box(0,10000,shape=(5,)), action_space=gym.spaces.Box(0,1,shape=(3,)), config=HRS_config) for i in range(args.num_HRS)}
    # policies["HDS"] = PolicySpec(observation_space=gym.spaces.Box(0,10000,shape=(2,3)), action_space=gym.spaces.Box(0,1,shape=(3,)), config=HDS_config)
    # print(policies)
    # policies = {
    #     "HDS": PolicySpec(policy_class=sac.SACTFPolicy, observation_space=gym.spaces.Box(0, 10000, shape=(2, 3)),
    #                       action_space=gym.spaces.Box(0, 1, shape=(3,)), config=HDS_config),
    #     "HRS0": PolicySpec(policy_class=ppo.PPOTFPolicy, observation_space=gym.spaces.Box(0, 10000, shape=(5,)),
    #                        action_space=gym.spaces.Box(0, 1, shape=(3,)), config=HRS0_config),
    #     "HRS1": PolicySpec(policy_class=ppo.PPOTFPolicy, observation_space=gym.spaces.Box(0, 10000, shape=(5,)),
    #                        action_space=gym.spaces.Box(0, 1, shape=(3,)), config=HRS1_config)
    # },

    def policy_mapping_fn(agent_id, episdoe, worker, **kwargs):
        if agent_id.startswith("HRS"):
            policy_id = agent_id[:4]
            # print(policy_id)
            return policy_id
        else:
            return agent_id

    algo = HyperOptSearch()
    # algo = ConcurrencyLimiter(algo, max_concurrent=24)
    scheduler = AsyncHyperBandScheduler()

    tune.run(
        "PPO",
        local_dir=f'~/MARL_results/extend_scenario_tune/trial{trial}',
        # stop=stop,
        config={
            "env": HydroRefuelSys,
            "env_config": {
              "num_HRS": args.num_HRS,
                # "Pmax": [250, 150]
                # "Hmax": [5,6]
                # "SOHmax": [30, 20]
                # "DGmax": [50, 0],
                "ELEmax": [250, 450],
                # "Hpie_max": [3, 10]
            },
            "disable_env_checking": True,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
            "num_workers": 1,
            # "lr": 1e-6,
            "multiagent": {
                # Use a simple set of policy IDs. Spaces for the individual policies
                # will be inferred automatically using reverse lookup via the
                # `policy_mapping_fn` and the env provided spaces for the different
                # agents. Alternatively, you could use:
                # policies: {main0: PolicySpec(...), main1: PolicySpec}
                # "policies": policies,
                "policies": {
                    "HDS": PolicySpec(observation_space=gym.spaces.Box(0, 10000, shape=(2,3)), action_space=gym.spaces.Box(0, 1, shape=(3,)), config=HDS_config),
                    "HRS0": PolicySpec(observation_space=gym.spaces.Box(0, 10000, shape=(5,)), action_space=gym.spaces.Box(0, 1, shape=(3,)), config=HRS0_config),
                    "HRS1": PolicySpec(observation_space=gym.spaces.Box(0, 10000, shape=(5,)), action_space=gym.spaces.Box(0, 1, shape=(3,)), config=HRS1_config)
                },
                # Simple mapping fn, mapping agent0 to main0 and agent1 to main1.
                # "policy_mapping_fn": (
                #     lambda aid, episode, worker, **kw: f"main{aid[1]}"
                # ),
                "policy_mapping_fn": policy_mapping_fn,
                # Only train main0.
                # "policies_to_train": ["mainD"],
            },
            "framework": args.framework,
            # "eager_tracing": args.eager_tracing,
        },
        metric = "episode_reward_mean",
        mode = "max",
        search_alg = algo,
        scheduler = scheduler,
        num_samples = 1500
    )
