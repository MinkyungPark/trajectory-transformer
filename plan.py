import os
import argparse
from posixpath import split
import dill
import json

import torch
import numpy as np

from model import GPT
from dataset import load_environment
from search import make_prefix, beam_plan, extract_actions, update_context
from utils import check_dir

parser = argparse.ArgumentParser()

parser.add_argument('--plan_freq', type=int, default=1)
parser.add_argument('--horizon', type=int, default=15)
parser.add_argument('--beam_width', type=int, default=128)
parser.add_argument('--n_expand', type=int, default=2)
parser.add_argument('--k_obs', type=int, default=1)
parser.add_argument('--k_act', type=int, default=None)
parser.add_argument('--cdf_obs', type=int, default=None)
parser.add_argument('--cdf_act', type=int, default=0.6)
parser.add_argument('--percentile', type=str, default='mean')
parser.add_argument('--max_context_transitions', type=int, default=5)
parser.add_argument('--prefix_context', type=bool, default=True)
parser.add_argument('--loadpath', type=str, default='logs')
parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
parser.add_argument('--mode', type=str, default='base')
parser.add_argument('--model_epoch', type=int, default=48)

args = parser.parse_args()

loadpath = os.path.join(args.loadpath, args.dataset, args.mode)

file_list = os.listdir(loadpath)
file_list = [f for f in file_list if 'state' in f]
max_epoch = 0
for f in file_list:
    splited = f.split('_')[1]
    splited = int(splited.split('.')[0])
    if splited > max_epoch:
        max_epoch = splited
args.model_epoch = max_epoch

print(f'Dataset [{args.dataset}] / Mode : [{args.mode}]')

# dconf = pickle.load(open(loadpath + '/data_config.pkl', 'rb'))
dconf = dill.load(open(loadpath + '/data_config.dill', 'rb'))
discretizer = dconf.discretizer
discount = dconf.discount
# s_dim = dconf.observation_dim
# a_dim = dconf.action_dim
try:
    s_dim = dconf.s_dim
    a_dim = dconf.a_dim
except:
    s_dim = dconf.observation_dim
    a_dim = dconf.action_dim

# mconf = pickle.load(open(loadpath + '/model_config.pkl', 'rb'))
mconf = dill.load(open(loadpath + '/model_config.dill', 'rb'))
model = GPT(mconf)
model.load_state_dict(torch.load(loadpath + '/state_'+ str(args.model_epoch) +'.pt'))
model.to(torch.device('cuda'))

value_fn = lambda x: discretizer.value_fn(x, args.percentile)

env = load_environment(args.dataset)
observation = env.reset()
total_reward = 0

# tokenized transitions for conditioning transformer
context = []

def recon_index(obs=np.array([]), act=np.array([]), type=None):
    if not type or type == 'reconstruct':
        idx = dconf.shuff_ind
        if obs.size > 0:
            return obs[idx[ : s_dim]]
        if act.size > 0:
            idx = idx[s_dim : s_dim + a_dim] - s_dim
            recon_idx = np.array([np.where(idx == i)[0][0] for i in range(idx.size)])
            return act[recon_idx]
    elif type == 'reverse':
        if obs.size > 0:
            return np.flip(obs)
        if act.size > 0:
            return np.flip(act)
    else:
        pass # error


T = env.max_episode_steps
for t in range(T):
    if t % args.plan_freq == 0:
        ## concatenate previous transitions and current observations to input to model
        if args.mode == 'shuffle':
            observation = recon_index(obs=observation)
        if args.mode == 'reverse':
            observation = recon_index(obs=observation, type='reverse')
        
        prefix = make_prefix(discretizer, context, observation, args.prefix_context)

        ## sample sequence from model beginning with `prefix`
        sequence = beam_plan(
            model, value_fn, prefix,
            args.horizon, args.beam_width, args.n_expand, s_dim, a_dim,
            discount, args.max_context_transitions,
            k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
        )

    else:
        sequence = sequence[1:]

    ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
    sequence_recon = discretizer.reconstruct(sequence)

    ## [ action_dim ] index into sampled trajectory to grab first action
    action = extract_actions(sequence_recon, s_dim, a_dim, t=0)

    if args.mode == 'shuffle':
        step_action = recon_index(act=action, type='reconstruct') 
    elif args.mode == 'reverse':
        step_action = recon_index(act=action, type='reverse')
    else:
        step_action = action

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(step_action)

    ## update return
    total_reward += reward
    score = env.get_normalized_score(total_reward)

    ## update rollout observations and context transitions
    context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions)

    print(
        f'[ plan ] timestep: {t} / {T} | reward: {reward:.2f} | total Reward: {total_reward:.2f} | normalized_score: {score:.4f} | \n'
    )

    if terminal: break

    observation = next_observation


json_path = check_dir(os.path.join(loadpath, 'plan'))
json_data = {'normalized_score': score, 'step': t, 'total_return': total_reward, 'terminal': terminal, 'gpt_epoch': args.model_epoch}
json.dump(json_data, open(json_path + '/rollout.json', 'w'), indent=2, sort_keys=True)