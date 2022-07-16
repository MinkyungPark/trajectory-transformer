import numpy as np
from search import make_prefix, beam_plan, extract_actions, update_context


def recon_index(obs=np.array([]), act=np.array([]), type=None, shuff_ind=None, s_dim=0, a_dim=0):
    if not type or type == 'reconstruct':
        idx = shuff_ind
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


def plan(s_dim, a_dim, mode, plan_freq, discretizer,
        shuff_ind, prefix_context, model, horizon, beam_width, n_expand,
        discount, max_context_transitions, env,
        k_obs=1, k_act=None, cdf_obs=None, cdf_act=0.6, percentile='mean', T=None):

    observation = env.reset()
    total_reward = 0
    context = []
    value_fn = lambda x: discretizer.value_fn(x, percentile)
    if not T: T = env.max_episode_steps

    for t in range(T):
        if t % plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            if mode == 'shuffle':
                observation = recon_index(obs=observation, shuff_ind=shuff_ind, s_dim=s_dim, a_dim=a_dim)
            if mode == 'reverse':
                observation = recon_index(obs=observation, type='reverse')
            
            prefix = make_prefix(discretizer, context, observation, prefix_context)

            ## sample sequence from model beginning with `prefix`
            sequence = beam_plan(
                model, value_fn, prefix,
                horizon, beam_width, n_expand, s_dim, a_dim,
                discount, max_context_transitions,
                k_obs=k_obs, k_act=k_act, cdf_obs=cdf_obs, cdf_act=cdf_act,
            )
        else:
            sequence = sequence[1:]

        ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
        sequence_recon = discretizer.reconstruct(sequence)

        ## [ action_dim ] index into sampled trajectory to grab first action
        action = extract_actions(sequence_recon, s_dim, a_dim, t=0)

        if mode == 'shuffle':
            step_action = recon_index(act=action, type='reconstruct', shuff_ind=shuff_ind, s_dim=s_dim, a_dim=a_dim) 
        elif mode == 'reverse':
            step_action = recon_index(act=action, type='reverse')
        else:
            step_action = action

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(step_action)

        ## update return
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        ## update rollout observations and context transitions
        context = update_context(context, discretizer, observation, action, reward, max_context_transitions)

        print(
            f'[ plan ] timestep: {t} / {T} | reward: {reward:.2f} | total Reward: {total_reward:.2f} | normalized_score: {score:.4f} | \n'
        )

        if terminal: break

        observation = next_observation

    return score, t, total_reward, terminal