import os
import argparse
import dill
import json

import torch
from model import GPT
from dataset import load_environment
from utils import check_dir
from plan import plan

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

env = load_environment(args.dataset)

score, t, total_reward, terminal = plan(s_dim, a_dim, args.mode, args.plan_freq, discretizer, 
        args.prefix_context, model, args.horizon, args.beam_width, args.n_expand,
        discount, args.max_context_transitions, env,
        args.k_obs, args.k_act, args.cdf_obs, args.cdf_act, args.percentile)

json_path = check_dir(os.path.join(loadpath, 'plan'))
json_data = {'normalized_score': score, 'step': t, 'total_return': total_reward, 'terminal': terminal, 'gpt_epoch': args.model_epoch}
json.dump(json_data, open(json_path + '/rollout.json', 'w'), indent=2, sort_keys=True)