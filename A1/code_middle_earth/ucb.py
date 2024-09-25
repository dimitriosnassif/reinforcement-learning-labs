from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots

def ucb(
    heroes: Heroes, 
    c: float, 
    init_value: float = .0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Upper Confidence Bound (UCB) action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param c: The exploration coefficient that balances exploration vs. exploitation.
    :param init_value: Initial estimation of each hero's value.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: TThe average of rewards up to step t. For example: If 
    we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """

    num_heroes = len(heroes.heroes)
    values = [init_value] * num_heroes   # Initial action values
    counts = [0] * num_heroes
    rew_record = []                      # Rewards at each timestep
    avg_ret_record = []                  # Average reward up to each timestep
    tot_reg_record = []                  # Total regret up to each timestep
    opt_action_record = []               # Percentage of optimal actions selected
    
    total_rewards = 0
    total_regret = 0

    true_probabilities = [hero['true_success_probability'] for hero in heroes.heroes]
    optimal_hero_index = np.argmax(true_probabilities)
    optimal_reward = true_probabilities[optimal_hero_index]

    for t in range(1, heroes.total_quests + 1):
        if t <= num_heroes:
            chosen_hero = t - 1
        else:
            ucb_values = [
                values[i] + c * np.sqrt(np.log(t) / counts[i]) if counts[i] > 0 else float('inf')
                for i in range(num_heroes)
            ]
            chosen_hero = np.argmax(ucb_values)
        
        reward = heroes.attempt_quest(chosen_hero)

        total_rewards += reward
        regret = optimal_reward - reward
        total_regret += regret
        rew_record.append(reward)
        avg_ret_record.append(total_rewards / t)
        tot_reg_record.append(total_regret)

        is_optimal_action = (chosen_hero == optimal_hero_index)
        if t == 1:
            opt_action_record.append(1 if is_optimal_action else 0)
        else:
            opt_action_record.append((opt_action_record[-1] * t + (1 if is_optimal_action else 0)) / (t + 1))
        
        counts[chosen_hero] += 1
        values[chosen_hero] += (reward - values[chosen_hero]) / counts[chosen_hero]
    
    return rew_record, avg_ret_record, tot_reg_record, opt_action_record

if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various c values
    c_values = [0.0, 0.5, 2.0]
    results_list = []
    for c in c_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                    heroes=heroes, bandit_method=ucb, 
                                                                    c=c, init_value=0.0)
        
        results_list.append({
            'exp_name': f'c={c}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })

    save_results_plots(results_list, plot_title='UCB Experiment Results On Various C Values', 
                       results_folder='results', pdf_name='ucb_various_c_values.pdf')
