from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots

def softmax(x, tau=1):
    """ Returns softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: idx -- chosen index
    """
    
    e_x = np.exp(np.array(x) / tau)
    return e_x / e_x.sum(axis=0)


def gradient_bandit(
    heroes: Heroes, 
    alpha: float, 
    use_baseline: bool = True,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Gradient Bandit action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param alpha: The learning rate.
    :param use_baseline: Whether or not use avg return as baseline.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: TThe average of rewards up to step t. For example: If 
    we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """

    num_heroes = len(heroes.heroes)
    h = np.array([0]*num_heroes, dtype=float)  # init h (the logits)
    rew_record = []                            # Rewards at each timestep
    avg_ret_record = []                        # Average reward up to each timestep
    tot_reg_record = []                        # Total regret up to each timestep
    opt_action_record = []                     # Percentage of optimal actions selected
    
    reward_bar = 0
    total_rewards = 0
    total_regret = 0

    true_probabilities = [hero['true_success_probability'] for hero in heroes.heroes]
    optimal_hero_index = np.argmax(true_probabilities)
    optimal_reward = true_probabilities[optimal_hero_index]

    for t in range(heroes.total_quests):
        probs = softmax(h)
        chosen_hero = np.random.choice(num_heroes, p=probs)
        reward = heroes.attempt_quest(chosen_hero)

        total_rewards += reward
        regret = optimal_reward - reward
        total_regret += regret

        rew_record.append(reward)
        avg_ret_record.append(total_rewards / (t + 1))
        tot_reg_record.append(total_regret)

        is_optimal_action = (chosen_hero == optimal_hero_index)
        if t == 0:
            opt_action_record.append(1 if is_optimal_action else 0)
        else:
            opt_action_record.append((opt_action_record[-1] * t + (1 if is_optimal_action else 0)) / (t + 1))

        if use_baseline:
            reward_bar += (reward - reward_bar) / (t + 1)

        for i in range(num_heroes):
            if i == chosen_hero:
                h[i] += alpha * (reward - (reward_bar if use_baseline else 0)) * (1 - probs[i])
            else:
                h[i] -= alpha * (reward - (reward_bar if use_baseline else 0)) * probs[i]
    
    return rew_record, avg_ret_record, tot_reg_record, opt_action_record

if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various alpha values with baseline
    alpha_values = [0.05, 0.1, 2]
    results_list = []
    for alpha in alpha_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=gradient_bandit,
                                                                    alpha=alpha, use_baseline=True)
        results_list.append({
            "exp_name": f"alpha={alpha}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })
    
    save_results_plots(results_list, plot_title="Gradient Bandits (with Baseline) Experiment Results On Various Alpha Values",
                       results_folder='results', pdf_name='gradient_bandit_various_alpha_values_with_baseline.pdf')

    # Test various alpha values without baseline
    alpha_values = [0.05, 0.1, 2]
    results_list = []
    for alpha in alpha_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=gradient_bandit,
                                                                    alpha=alpha, use_baseline=False)
        results_list.append({
            "exp_name": f"alpha={alpha}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })

    save_results_plots(results_list, plot_title="Gradient Bandits (without Baseline) Experiment Results On Various Alpha Values",
                       results_folder='results', pdf_name='gradient_bandit_various_alpha_values_without_baseline.pdf')
