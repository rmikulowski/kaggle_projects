import numpy as np
import pandas as pd
import os
os.chdir('./rock_paper_scissors')

import matplotlib.pyplot as plt
from kaggle_environments import make, evaluate

if __name__ == "__main__":

    list_names = [
        'self_popular_12pattern',
        'timestep_self_popular_12pattern'
        ]

    list_agents = [agent_name + ".py" for agent_name in list_names]

    scores = np.zeros((len(list_names), len(list_names)), dtype=int)
    min_scores = np.zeros((len(list_names), len(list_names)), dtype=int)
    max_scores = np.zeros((len(list_names), len(list_names)), dtype=int)
    
    # Create environment
    ENV_NAME = 'rps'
    NR_STEPS = 500
    env = make(ENV_NAME, configuration={"episodeSteps": NR_STEPS})

    for i in (list_agents):
        if not (os.path.exists(i)):
            raise ImportError('One of the agents path is not well defined')


    for ind_agent_1 in range(len(list_names)):
        for ind_agent_2 in range(ind_agent_1 + 1, len(list_names)):
            print(f"LOG: {list_names[ind_agent_1]} vs {list_names[ind_agent_2]}", end="\r")
            
            results = []
            for i in range(3):
                current_score = evaluate(
                    ENV_NAME, 
                    [list_agents[ind_agent_1], list_agents[ind_agent_2]], 
                    configuration={"episodeSteps": NR_STEPS}
                )
                if current_score[0][0] is None:
                    results.append(0)
                else:
                    results.append(current_score[0][0])

            
            scores[ind_agent_1, ind_agent_2] = np.mean(results)
            scores[ind_agent_2, ind_agent_1] = -np.mean(results)
            min_scores[ind_agent_1, ind_agent_2] = min(results)
            min_scores[ind_agent_2, ind_agent_1] = -min(results)
            max_scores[ind_agent_1, ind_agent_2] = max(results)
            max_scores[ind_agent_2, ind_agent_1] = -max(results)
        
        print()

    pd.DataFrame(scores, index = list_names, columns = list_names).to_csv('scores_mean.csv')
    pd.DataFrame(min_scores, index = list_names, columns = list_names).to_csv('scores_min.csv')    
    pd.DataFrame(max_scores, index = list_names, columns = list_names).to_csv('scores_max.csv')
