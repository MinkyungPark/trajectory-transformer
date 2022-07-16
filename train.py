import os
import math
import argparse
import datetime
import dill
import json

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import GPT, GPTConfig
from dataset import DiscretizedDataset, load_environment
from utils import set_seed, check_dir, to, Timer
from plan import plan


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='halfcheetah-medium-expert-v2')
parser.add_argument('--max_path_length', type=int, default=1000)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--n_head', type=int, default=4)

parser.add_argument('--n_epochs_ref', type=int, default=50)
parser.add_argument('--n_saves', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda:3')

parser.add_argument('--n_embd', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=6e-4)
parser.add_argument('--lr_decay', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)

parser.add_argument('--step', type=int, default=1)
parser.add_argument('--subsampled_sequence_length', type=int, default=10)
parser.add_argument('--termination_penalty', type=int, default=-100)

parser.add_argument('--discretizer', type=str, default='QuantileDiscretizer')
parser.add_argument('--action_weight', type=int, default=5)
parser.add_argument('--reward_weight', type=int, default=1)
parser.add_argument('--value_weight', type=int, default=1)

args = parser.parse_args()
set_seed(args.seed)
savepath = check_dir(os.path.join('logs', args.dataset, 'base'))


################# Dataset #################

sequence_length = args.subsampled_sequence_length * args.step

dataset = DiscretizedDataset(
    savepath=savepath,
    env=args.dataset,
    N=args.N,
    penalty=args.termination_penalty,
    sequence_length=sequence_length,
    step=args.step,
    discount=args.discount,
    max_path_length=args.max_path_length,
)

obs_dim = dataset.s_dim
act_dim = dataset.a_dim
transition_dim = dataset.joined_dim
block_size = args.subsampled_sequence_length * transition_dim - 1


################# Model GPT #################

mconf = GPTConfig(vocab_size=args.N, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd*args.n_head,
    action_weight=args.action_weight, reward_weight=args.reward_weight, value_weight=args.value_weight,
    embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
    block_size=block_size, observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim)

model = GPT(config=mconf)
dill.dump(mconf, open(savepath + '/model_config.dill', 'wb'))
model.to(args.device)


################# Trainer #################

class Trainer:
    def __init__(self, savepath, **kwargs):
        self.n_epochs = 0
        self.n_tokens = 0 # counter used for learning rate decay
        self.optimizer = None

        for k,v in kwargs.items():
            setattr(self, k, v)
        dill.dump(kwargs, open(savepath + '/trainer_config.dill', 'wb'))
        
        self.writer = SummaryWriter(savepath)
        self.json_data = []

    def get_optimizer(self, model):
        if self.optimizer is None:
            self.optimizer = model.configure_optimizers(self)
        return self.optimizer

    def model_parr(self, model):
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(model).to(self.device)

    def train(self, model, dataset, n_epochs=1, log_freq=100):
        self.model_parr(model)
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = self.get_optimizer(raw_model)
        # optimizer = self.get_optimizer(model)

        model.train(True)
        vocab_size = dataset.N

        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)

        for _ in range(n_epochs):
            losses = []
            timer = Timer()
            for it, batch in enumerate(loader):
                batch = to(batch, self.device)
                with torch.set_grad_enabled(True):
                    logits, loss = model(*batch)
                    loss = loss.mean()
                    losses.append(loss.item())

                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm_clip)
                optimizer.step()

                # decay the learning rate based on our progress
                if self.lr_decay:
                    y = batch[-2]
                    self.n_tokens += (y != vocab_size).sum() # number of tokens processed this step
                    if self.n_tokens < self.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.n_tokens) / float(max(1, self.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.n_tokens - self.warmup_tokens) / float(max(1, self.final_tokens - self.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.learning_rate

                if it % log_freq == 0:
                    print(
                        f'[ training ] epoch {self.n_epochs} [ {it:4d} / {len(loader):4d} ] ',
                        f'train loss {loss.item():.5f} | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                        f't: {timer():.2f}')

            self.n_epochs += 1

        # eval
        total_returns = []
        for i in range(3):
            score, t, total_reward, terminal = plan(obs_dim, act_dim, plan_freq=1, discretizer=dataset.discretizer, 
                prefix_context=True, model=model.module, horizon=15, beam_width=128, n_expand=2,
                discount=dataset.discount, max_context_transitions=5, env=load_environment(args.dataset), T=500)
            total_returns.append(total_reward)
            
            self.json_data.append({'normalized_score': score, 'step': t, 'total_return': total_reward, 'terminal': terminal, 'epoch': self.n_epochs})

        self.writer.add_scalar('Train/eval_return', np.mean(total_returns), self.n_epochs)
        self.writer.add_scalar('Train/loss', loss.item(), self.n_epochs)



warmup_tokens = len(dataset) * block_size ## number of tokens seen per epoch
final_tokens = 20 * warmup_tokens

trainer = Trainer(
    savepath=savepath,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1, # only applied on matmul weights
    lr_decay=args.lr_decay,
    warmup_tokens=warmup_tokens,
    final_tokens=final_tokens,
    num_workers=4,
    # device=args.device,
)

################# Training ! #################

tr_stt = datetime.datetime.now()

n_epochs = int(1e6 / len(dataset) * args.n_epochs_ref) + 10
save_freq = int(n_epochs // args.n_saves)

for epoch in range(n_epochs):
    print(f'\nEpoch: {epoch} / {n_epochs} | {args.dataset} ')
    trainer.train(model, dataset)
    save_epoch = (epoch + 1) // save_freq * save_freq
    statepath = os.path.join(savepath, f'state_{save_epoch}.pt')
    
    print(f'Saving model to {statepath}')
    state = model.state_dict()
    torch.save(state, statepath)

json.dump(trainer.json_data, open(savepath + '/rollout_eval.json', 'w'), indent=2, sort_keys=True)
print(f'training start-time: {tr_stt} | training end-time : {datetime.datetime.now()}')