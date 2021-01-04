import argparse
import gym
import numpy as np
import os
import torch
import csv

from models import TD3
from utils import memory
from gym.envs.registration import register
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import clear_output
from IPython.display import display


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=1, test=True, episode_num=0):
#    policy.eval_mode()
    avg_reward = 0.
    eval_steps = 0
    avg_steps = 0
    env = gym.make(env_name)
    env.seed(seed + 100)
    episode_steps = 0
    max_episode_steps = 480
    bg_act = []

    for _ in range(eval_episodes):
#        if test:
#            env.render(mode='human', close=False)
        state, done = env.reset(), False
        hidden = None
        while not done:
            if test:
                env.render(mode='human', close=False)
            action, hidden = policy.select_action(np.array(state), hidden)
            # env.render(mode='human', close=False)
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            episode_steps = episode_steps+1
            if episode_steps >= max_episode_steps:
                done = True
#            bg_act.append([state[59], action, reward, done])
        eval_steps = eval_steps + episode_steps
#    print(bg_act)
    avg_reward /= eval_episodes
    avg_steps = eval_steps / eval_episodes

#    policy.train_mode()
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} Agerage steps {avg_steps}")
    print("---------------------------------------")
    plt.savefig(f"./figures/td3_eval_{episode_num}.png")
    return avg_reward, avg_steps

def plot(rewards, steps, episode):
    clear_output(True)
    plt.figure(figsize=(30,10))
    plt.plot(rewards)
    plt.plot(steps)
    plt.savefig(f"./figures/td3_lstm_{episode}.png")
    # plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")

    seed = 0
    parser.add_argument("--start_timesteps", default=5e3, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=1e3, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=6e5, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.05)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=100, type=int)
    # Memory size
    parser.add_argument("--memory_size", default=2e4, type=int)
    # Learning rate
    parser.add_argument("--lr", default=3e-2, type=float)
    # Discount factor
    # Model width
    parser.add_argument("--hidden_size", default=300, type=int)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.01)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.005)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Use recurrent policies or not
    parser.add_argument("--save_model", action="store_true")
#    # Model load file name, "" doesn't load, "default" uses file_name#
    parser.add_argument("--load_model", default="")
#    parser.add_argument("--load_model", default="store_true")
    # Don't train and just run the model
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    max_episode_steps = 480
    model_filename = 'TD3_child001'

    env_name = "simglucose.envs:simglucose-v0"  # Name of a environment (set it to any Continous environment you want)


    register(
        id=env_name,
        entry_point='simglucose.envs:T1DSimEnv',
        max_episode_steps=480,
        kwargs={'patient_name': 'child#001', 'reward_fun': 'magni_reward'}
    )


    file_name = f"{args.policy}_{env_name}_{seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./npy_results"):
            os.makedirs("./npy_results")

    if not os.path.exists("./figures"):
            os.makedirs("./figures")

#    if args.save_model and not os.path.exists("./models"):
    if not os.path.exists("./models"):
            os.makedirs("./models")


    env = gym.make('simglucose.envs:simglucose-v0')

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    recurrent_actor = False
    recurrent_critic = False

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": args.hidden_size,
        "discount": 0.99,
        "tau": 0.005,
        "recurrent_actor": recurrent_actor,
        "recurrent_critic": recurrent_critic,
    }

    # Initialize policy

    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)


    if args.load_model != "":
#        policy_file = file_name \
#            if args.load_model == "default" else args.load_model
        policy.load("./models/TD3_child001")

    if args.test:
        eval_policy(policy, args.env, args.seed, eval_episodes=4, test=True)
        return

    replay_buffer = memory.ReplayBuffer(
        state_dim, action_dim, args.hidden_size,
        args.memory_size, recurrent=recurrent_actor)

    evaluations = []
    eval_steps = []
    # Evaluate untrained policy
    eval_rew, eval_step = eval_policy(policy, env_name, seed, episode_num=0)
    eval_steps.append(eval_step)
    evaluations.append(eval_rew)

#    evaluations = [eval_policy(policy, env_name, seed)]
    best_reward = 0
 #   best_reward = evaluations[-1]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 1
    episode_results = []
    hidden = policy.get_initial_states()

    for t in range(1, int(args.max_timesteps)):

        episode_timesteps += 1
        bg_act = []
        if episode_num % 10 == 0:
            env.render(mode='human', close=False)

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
            _, next_hidden = policy.select_action(np.array(state), hidden)
        else:
            a, next_hidden = policy.select_action(np.array(state), hidden)
            action = (
                a + np.random.normal(
                    0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
#        bg_act.append([state[59],action[-1],next_state[59],reward, done])
        done_bool = float(
            done) if episode_timesteps < max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(
            state, action, next_state, reward, done_bool, hidden, next_hidden)

        state = next_state
        hidden = next_hidden
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it
            #  will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num} "
                f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            episode_result = [episode_num,t+1,episode_timesteps, episode_reward]
            episode_results.append(episode_result)
            if episode_num % 10 == 0:
                plt.savefig(f"./figures/td3_train_{episode_num}.png")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            hidden = policy.get_initial_states()

        # Evaluate episode
#        if (t + 1) % args.eval_freq == 0:
#        if done:
            if (episode_num % 20 == 0):
                eval_rew, eval_step = eval_policy(policy, env_name, seed, episode_num=episode_num)
                eval_steps.append(eval_step)
                evaluations.append(eval_rew)
                plot(evaluations, eval_steps, episode_num)
#            evaluations.append(eval_policy(policy, env_name, seed))
#            if evaluations[-1] > best_reward and args.save_model:
                if evaluations[-1] > best_reward:
                    policy.save(f"./models/{model_filename}")
                    best_reward = evaluations[-1]
                    print('Best Reward so far :', best_reward)
            np.save("./npy_results/td3_child001_rew", evaluations)
            np.save("./npy_results/td3_child001_steps", eval_steps)
            np.save("./npy_results/td3_child001_episode_results", episode_results)
#            if (t + 1) % (args.eval_freq*10) == 0:
    print('Prcess completed at', t, ' timesteps')

if __name__ == "__main__":
    main()
