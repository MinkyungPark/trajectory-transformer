import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataset import load_environment
from search import plan

        
        
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



    json.dump(trainer.json_data, open(savepath + '/rollout_eval.json', 'w'), indent=2, sort_keys=True)


        self.writer = SummaryWriter(savepath)
        self.json_data = []