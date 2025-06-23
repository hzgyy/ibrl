import argparse
import os
import time
import common_utils
import train_bc
import train_rl
import rich.traceback
import torch
import numpy as np
from env.robosuite_wrapper import PixelRobosuite
from common_utils import ibrl_utils as utils

def eval_kl(agent1,agent2,obs_batch,is_dict=0,stddev=0.1):
    if is_dict:
        pi1 = agent1.actor.torch_forward(obs_batch,stddev)
        pi2 = agent2.actor.torch_forward(obs_batch,stddev)
        #action1, log_pi1 = agent1.actor.act_with_prob(obs_batch,stddev)
        #log_pi2 = agent2.actor.logp(obs_batch,action1,stddev)
    else:
        pi1 = agent1.actor.torch_forward({'state':obs_batch},stddev)
        pi2 = agent2.actor.torch_forward({'state':obs_batch},stddev)
        # action1, log_pi1 = agent1.actor.act_with_prob({'state':obs_batch},stddev)
        # log_pi2 = agent2.actor.logp({'state':obs_batch},action1,stddev)
    #kl = (torch.exp(log_pi1)*(log_pi1-log_pi2)).mean()
    kl = torch.distributions.kl_divergence(pi1, pi2).mean()
    return kl.item()


if __name__ == "__main__":
    rich.traceback.install()
    # os.environ["MUJOCO_GL"] = "egl"
    # torch.backends.cudnn.benchmark = False  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore
    # np.set_printoptions(precision=4, linewidth=100, suppress=True)
    # torch.set_printoptions(linewidth=100, sci_mode=False)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--folder1", type=str, default=None)
    parser.add_argument("--include1", type=str, default="model0")
    parser.add_argument("--folder2", type=str, default=None)
    parser.add_argument("--include2", type=str, default="model0")
    parser.add_argument("--mode1", type=str, default="bc", help="bc/rl")
    parser.add_argument("--mode2", type=str, default="bc", help="bc/rl")
    parser.add_argument("--num_game", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    common_utils.set_all_seeds(args.seed)

    weights1 = common_utils.get_all_files(args.folder1, ".pt", args.include1)
    weights2 = common_utils.get_all_files(args.folder2, ".pt", args.include2)

    print(f"files to eval:")
    eval_items = []
    for weight in weights1:
        print(weight, f", repeat {args.repeat}")
        if args.mode1 == "bc":
            agent1, _, env_params1 = train_bc.load_model(weight, "cuda")
        elif args.mode1 == "rl":
            agent1, _, env_params1 = train_rl.load_model(weight, "cuda")
        else:
            assert False, f"unsupported mode: {args.mode1}"

        for _ in range(args.repeat):
            eval_items.append((weight, agent1, env_params1))
    for weight in weights2:
        print(weight, f", repeat {args.repeat}")
        if args.mode2 == "bc":
            agent2, _, env_params2 = train_bc.load_model(weight, "cuda")
        elif args.mode2 == "rl":
            agent2, _, env_params2 = train_rl.load_model(weight, "cuda")
        else:
            assert False, f"unsupported mode: {args.mode2}"

        for _ in range(args.repeat):
            eval_items.append((weight, agent2, env_params2))
    print(common_utils.wrap_ruler(""))
    
    #collect observations from rollouts
    states = []
    for _,agent,env_params in eval_items:
        env = PixelRobosuite(**env_params)
        with torch.no_grad(), utils.eval_mode(agent):
            for episode_idx in range(args.num_game):
                rewards = []
                np.random.seed(args.seed + episode_idx)
                obs, _ = env.reset()
                states.append(obs["state"])
                terminal = False
                while not terminal:
                    action = agent.act(obs, eval_mode=True,stddev=0.05)
                    obs, reward, terminal, _, image_obs = env.step(action)
                    states.append(obs["state"])
    assert len(states) >= args.num_samples, "states number less than num_samples"
    states = torch.stack(states,dim=0)
    print(f"finish data collection, states number:{states.shape[0]}")
    indices = torch.randperm(states.shape[0])[:args.num_samples]
    batch = states[indices]
    kl = eval_kl(agent1,agent2,batch)
    print(f"kl(pi1|pi2) = {kl}!!!!!!!!!!!!!")



    
    


    