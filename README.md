# Autonomous Pacman
This repository contains an AI agent for Pacman in a multi-agent environment. Algorithms such as value iteration based on the Markov Decision Process as well as Q-learning is utilized for the creation of the AI agent to allow Pacman to navigate to the terminal state with the highest rewards while avoiding the ghosts scattered on the map.

## Testing the AI Pacman based on value iteration 
To test:
```shell
python pacman.py -l layouts/q1b_tinyMaze.lay -p Q1Agent -a discount=gamma,iterations=K -g StationaryGhost -n 20
```

Description of the parameters are as follows:
- `gamma`: Discount factor that ranges from 0 to 1
- `K`: Number of iterations

 The layouts that can be used to test are the layouts that has the prefix `q1b_`.

 The table below shows the parameter values used for the two parameters above that results in the best average score for each layout:
 | Layout | Discount Factor(gamma) | Number of Iterations(K) | 
 | --- | --- | --- | 
 | bigMaze | 0.7000 | 800 |
 | bigMaze2 | 0.7500 | 800 |
 | contoursMaze | 0.1000 | 400 |
 | mediumMaze | 0.9990 | 600 |
 | mediumMaze2 | 0.9990 | 550 |
 | openMaze | 0.9999 | 500 |
 | smallMaze | 0.9990 | 500 |
 | testMaze | 0.3000 | 200 | 
 | tinyMaze | 0.5000 | 300 |
 | trickyMaze | 0.9990 | 400 |

 ## Testing the AI Pacman based on Q-learning
 To test:
 ```shell
python pacman.py -l layouts/q2b_tinyMaze.lay -p Q2Agent -a epsilon=epsilon_value,gamma=gamma_value,alpha=alpha_value -x <K> -n <K+1> -g StationaryGhost
```

Description of the parameters are as follows:
- `epsilon_value`: Epsilon-Greedy value for Q-learning that ranges from 0 to 1
- `gamma_value`: Discount factor that ranges from 0 to 1
- `alpha_value`: Learning rate for Q-learning that ranges from 0 to 1
- `K`: Number of iterations

The layouts that can be used to test are the layouts that has the prefix `q2b_`.

The table below shows the parameter values used for the four parameters above that results in the best average score for each layout:
| Layout | Epsilon-Greedy(epsilon_value) | Learning Rate(alpha_value) | Discount Factor(gamma_value) | Number of Iterations(K) |
| --- | --- | --- | --- | --- |
| bigMaze | 0.85 | 0.700 | 0.70 | 500 | 
| bigMaze2 | 0.80 | 0.700 | 0.85 | 700 | 
| contoursMaze | 0.95 | 0.700 | 0.70 | 400 | 
| mediumMaze | 0.80 | 0.700 | 0.70 | 600 | 
| openMaze | 0.80 | 0.999 | 0.70 | 700 |
| smallMaze | 0.80 | 0.700 | 0.80 | 500 |
| smallMaze2 | 0.70 | 0.700 | 0.85 | 500 | 
| testMaze | 0.40 | 0.700 | 0.80 | 200 | 
| tinyMaze | 0.65 | 0.500 | 0.70 | 200 |
| trickyMaze | 0.70 | 0.850 | 0.70 | 300 |



