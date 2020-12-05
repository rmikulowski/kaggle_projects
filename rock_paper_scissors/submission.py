import pandas as pd
import numpy as np
import json

# base class for all agents, random agent
class agent():
    def initial_step(self):
        return np.random.randint(3)
    
    def history_step(self, history):
        return np.random.randint(3)
    
    def step(self, history):
        if len(history) == 0:
            return int(self.initial_step())
        else:
            return int(self.history_step(history))

#----------------------------------------- Other Agents -----------------------------------------#

# agent that returns (previousCompetitorStep + shift) % 3
class mirror_shift(agent):
    def __init__(self, shift=0):
        self.shift = shift
    
    def history_step(self, history):
        return (history[-1]['competitorStep'] + self.shift) % 3
    
    
# agent that returns (previousPlayerStep + shift) % 3
class self_shift(agent):
    def __init__(self, shift=0):
        self.shift = shift
    
    def history_step(self, history):
        return (history[-1]['step'] + self.shift) % 3    


# agent that beats the most popular step of competitor
class popular_beater(agent):
    def history_step(self, history):
        counts = np.bincount([x['competitorStep'] for x in history])
        return (int(np.argmax(counts)) + 1) % 3

    
# agent that beats the agent that beats the most popular step of competitor
class anti_popular_beater(agent):
    def history_step(self, history):
        counts = np.bincount([x['step'] for x in history])
        return (int(np.argmax(counts)) + 2) % 3

class pattern_matching(agent):
    def __init__(self, steps = 3, deterministic = False, counter_strategy = False, init_value = 0.1, decay = 1):

        # Define strategy 
        self.deterministic = deterministic
        self.counter_strategy = counter_strategy
        if counter_strategy:
            self.step_type = 'step' 
        else:
            self.step_type = 'competitorStep'

        # Bandit parameters
        self.init_value = init_value
        self.decay = decay
        self.steps = steps
        
    def history_step(self, history):
        if len(history) < self.steps + 1:
            return self.initial_step()
        
        next_step_count = np.zeros(3) + self.init_value
        pattern = [history[i][self.step_type] for i in range(- self.steps, 0)]
        
        for i in range(len(history) - self.steps):
            next_step_count = (next_step_count - self.init_value)/self.decay + self.init_value
            current_pattern = [history[j][self.step_type] for j in range(i, i + self.steps)]
            if np.sum([pattern[j] == current_pattern[j] for j in range(self.steps)]) == self.steps:
                next_step_count[history[i + self.steps][self.step_type]] += 1
        
        if next_step_count.max() == self.init_value:
            return self.initial_step()
        
        if  self.deterministic:
            step = np.argmax(next_step_count)
        else:
            step = np.random.choice([0,1,2], p = next_step_count/next_step_count.sum())
        
        if self.counter_strategy:
            # we predict our step using transition matrix (as competitor can do) and beat probable competitor step
            return (step + 2) % 3 
        else:
            # we just predict competitors step and beat it
            return (step + 1) % 3


#----------------------------------------- Bandit -----------------------------------------#


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

def multi_armed_bandit_agent(observation, configuration):
    
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
    
    return log_step(step, history, best_agent)
