import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import torch
import numpy as np

from env.metaworld_wrapper import PixelMetaWorld
from common_utils import ibrl_utils as utils
from common_utils import Recorder

def run_eval(
    env: PixelMetaWorld,
    agent,
    num_game,
    seed,
    record_dir=None,
    verbose=True,
    eval_mode=True,
):
    recorder = None if record_dir is None else Recorder(record_dir)

    scores = []
    with torch.no_grad(), utils.eval_mode(agent):
        for episode_idx in range(num_game):
            step = 0
            rewards = []
            np.random.seed(seed + episode_idx)
            obs, image_obs = env.reset()

            terminal = False
            while not terminal:
                if recorder is not None:
                    recorder.add(image_obs)

                action = agent.act(obs, eval_mode=eval_mode).numpy()
                obs, reward, terminal, _, image_obs = env.step(action)
                rewards.append(reward)
                step += 1

            if verbose:
                print(
                    f"seed: {seed + episode_idx}, "
                    f"reward: {np.sum(rewards)}, len: {env.time_step}"
                )

            scores.append(np.sum(rewards))

            if recorder is not None:
                save_path = recorder.save(f"episode{episode_idx}")
                reward_path = f"{save_path}.reward.pkl"
                print(f"saving reward to {reward_path}")
                pickle.dump(rewards, open(reward_path, "wb"))

    if verbose:
        print(f"num game: {len(scores)}, seed: {seed}, score: {np.mean(scores)}")

    return scores

if __name__ == "__main__":
    import argparse
    import os
    import time
    import common_utils
    import train_bc_mw
    import train_rl_mw
    import rich.traceback

    # make logging more beautiful
    rich.traceback.install()

    os.environ["MUJOCO_GL"] = "egl"
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--include", type=str, default=None)
    parser.add_argument("--mode", type=str, default="bc", help="bc/rl")
    parser.add_argument("--num_game", type=int, default=10)
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--mp", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    common_utils.set_all_seeds(args.seed)

    if args.folder is None:
        weights = [args.weight]
    else:
        weights = common_utils.get_all_files(args.folder, ".pt", args.include)

    print(f"files to eval:")
    eval_items = []
    for weight in weights:
        print(weight, f", repeat {args.repeat}")
        if args.mode == "bc":
            agent, eval_env, env_params = train_bc_mw.load_model(weight, "cuda")
        elif args.mode == "rl":
            agent, eval_env = train_rl_mw.load_model(weight, "cuda")
        else:
            assert False, f"unsupported mode: {args.mode}"

        for _ in range(args.repeat):
            eval_items.append((weight, agent, eval_env))
    print(common_utils.wrap_ruler(""))

    # if args.record_dir:
    #     assert len(eval_items) == 1
    #     if env_params["env_name"] == "TwoArmTransport":
    #         env_params["camera_names"] = ["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"]
    #     else:
    #         env_params["camera_names"] = ["agentview", "robot0_eye_in_hand"]

    weight_scores = []
    all_scores = []
    for weight, agent, eval_env in eval_items:
        t = time.time()
        print(args.record_dir)
        if args.mp >= 1:
            assert args.record_dir is None
            scores = run_eval(
                eval_env, agent, args.num_game, args.mp, args.seed, verbose=args.verbose
            )
        else:
            scores = run_eval(
                eval_env,
                agent,
                args.num_game,
                args.seed,
                args.record_dir,
                verbose=args.verbose,
            )
        all_scores.append(scores)
        print(f"weight: {weight}")
        print(f"score: {np.mean(scores)}, time: {time.time() - t:.1f}")
        weight_scores.append((weight, np.mean(scores)))

    if len(weight_scores) > 1:
        weight_scores = sorted(weight_scores, key=lambda x: -x[1])
        scores = []
        for weight, score in weight_scores:
            print(f"{weight} -> {score}")
            scores.append(score)
        print(f"average score: {100 * np.mean(scores):.2f}")
        print(f"max score: {100 * scores[0]:.2f}")
        max_score_per_seed = np.array(all_scores).max(0)
        print(f"max over seed: {100 * np.mean(max_score_per_seed):.2f}")