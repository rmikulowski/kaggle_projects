import pandas as pd
import numpy as np
import json

from rock_paper_scissors.submission import *

agents = {
    'self_0': self_shift(0),
    'self_1': self_shift(1),  
    'self_2': self_shift(2),
    'popular_beater': popular_beater(),
    'anti_popular_beater': anti_popular_beater(),
    'pattern_matching_1': pattern_matching(7, False, False, decay = 1.001),
    'pattern_matching_2': pattern_matching(7, False, False, decay = 1.1),
    'pattern_matching_3': pattern_matching(3, False, False, decay = 1.001),
    'pattern_matching_4': pattern_matching(3, False, False, decay = 1.1),
    'pattern_matching_5': pattern_matching(5, False, False, decay = 1.001),
    'pattern_matching_6': pattern_matching(5, False, False, decay = 1.1),
    'pattern_matching_12': pattern_matching(7, True, False, decay = 1.001),
    'pattern_matching_22': pattern_matching(7, False, True, decay = 1.1),
    'pattern_matching_32': pattern_matching(3, True, False, decay = 1.001),
    'pattern_matching_42': pattern_matching(3, False, True, decay = 1.1),
    'pattern_matching_52': pattern_matching(5, True, False, decay = 1.001),
    'pattern_matching_62': pattern_matching(5, False, True, decay = 1.1)
}

history = []

# Bandit state
# Numbers representing weights on winning/losing
bandit_state = {k:[1,1] for k in agents.keys()}

def self_popular_12pattern_200(observation, configuration):
    
    # bandits' params
    step_size = 3 # how much we increase a and b 
    decay_rate = 1.05 # how much do we decay old historical data
    
    global history, bandit_state
    
    # Logging the steps in a .csv
    def log_step(step = None, history = None, agent = None, competitorStep = None, file = 'history.csv'):
        if step is None:
            step = np.random.randint(3)
        if history is None:
            history = []
        history.append({'step': step, 'competitorStep': competitorStep, 'agent': agent})
        if file is not None:
            pd.DataFrame(history).to_csv(file, index = False)
        return step
    
    def update_competitor_step(history, competitorStep):
        history[-1]['competitorStep'] = int(competitorStep)
        return history
    
    # load history
    if observation.step == 0:
        pass
    else:
        history = update_competitor_step(history, observation.lastOpponentAction)
        
        # updating bandit_state using the result of the previous step
        # we can update all states even those that were not used
        for name, agent in agents.items():
            agent_step = agent.step(history[:-1])
            # Update (decay) both numbers (states) of bandit
            bandit_state[name][1] = (bandit_state[name][1] - 1) / decay_rate + 1
            bandit_state[name][0] = (bandit_state[name][0] - 1) / decay_rate + 1
            
            # if lost
            if (history[-1]['competitorStep'] - agent_step) % 3 == 1:
                bandit_state[name][1] += step_size
            # if won
            elif (history[-1]['competitorStep'] - agent_step) % 3 == 2:
                bandit_state[name][0] += step_size
            # if tie
            else:
                bandit_state[name][0] += step_size/2
                bandit_state[name][1] += step_size/2
            
    # we can use it for analysis later
    with open('bandit.json', 'w') as outfile:
        json.dump(bandit_state, outfile)
    
    
    # generate random number from Beta distribution for each agent and select the most lucky one
    # based on history of games (win/lose ratio)
    best_proba = -1
    best_agent = None
    for k in bandit_state.keys():
        proba = np.random.beta(bandit_state[k][0],bandit_state[k][1])
        if proba > best_proba:
            best_proba = proba
            best_agent = k
        
    step = agents[best_agent].step(history)
    
    if observation.step > 800:
        return np.random.randit(3)
    
    else:
        return log_step(step, history, best_agent)
