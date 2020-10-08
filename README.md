# Monte Carlo Tree Search for OpenAI gym framework

General Python implementation of Monte Carlo Tree Search for the use with Open AI Gym environments.

The MCTS Algorithm is based on the one from [muzero-general](https://github.com/PatrickKorus/muzero-general) which is 
forked from [here](https://github.com/werner-duvaud/muzero-general).

This code was part of my Bachelor Thesis:

#### [An Evaluation of MCTS Methods for Continuous Control Tasks](https://www.dropbox.com/s/9acbtihfmagn7el/Bachelor_Thesis___An_Evaluation_of_MCTS_Methods_for_Continuous_Control_Tasks_FINAL.pdf?dl=0) 

The source code of the experiments covered by the thesis can be found [here](https://github.com/PatrickKorus/MCTSOC).

## Dependencies

Python 3.8 is used. Dependencies are mainly numpy and gym. Simply run:

```shell script
pip install -r requirements.txt
```

## How to use

This implementation follows the common agent-environment scheme. The environment is Wrapped by the Game class defined,
in `game.py`, which ensures that the game's state can be deep copied. The main Game implementations for usage with 
OpenAI gym environments are `DiscreteGymGame` and `ContinuousGymGame`. 

A simple example would be:

```python
import gym
from mcts_general.agent import MCTSAgent
from mcts_general.config import MCTSAgentConfig
from mcts_general.game import DiscreteGymGame


# configure agent
config = MCTSAgentConfig()
config.num_simulations = 200
agent = MCTSAgent(config)

# init game
game = DiscreteGymGame(env=gym.make('CartPole-v0'))
state = game.reset()
done = False
reward = 0

# run a trajectory
while not done:
    action = agent.step(game, state, reward, done)
    state, reward, done = game.step(action)
    
    # game.render()     # uncomment for environment rendering

game.close()
``` 

A continuous environment can be set up similarly. Note that you have to choose `mu` and `sigma` for (Gaussian Normal) 
sampling actions. Usually it is a good choice to start with `mu` being at the middle of your action space and `sigma` 
being half your action space. So for example for `Pendulum-v0` the action space is in [-2 ,2] hence a good choice to 
start with is `mu = 0.` and `sigma = 2.` 

Example for Continuous Control:

```python
import gym
from mcts_general.agent import ContinuousMCTSAgent
from mcts_general.config import MCTSContinuousAgentConfig
from mcts_general.game import ContinuousGymGame


# configure agent
config = MCTSContinuousAgentConfig()
agent = ContinuousMCTSAgent(config)

# init game
game = ContinuousGymGame(env=gym.make('Pendulum-v0'), mu=0., sigma=2.)
state = game.reset()
done = False
reward = 0

while not done:
    action = agent.step(game, state, reward, done)
    state, reward, done = game.step(action)
    game.render()

game.close()
```

## Features

Please have a look at the `game` package for using different time-discretization during planning, and what 
hyper parameters can be chosen in the `config` class. You might also find some useful `gym.Wrapper`s in 
`common/wrapper.py`. An extensive example on how to use this implementation for MCTS-research can be found in the
[thesis experiments](https://github.com/PatrickKorus/MCTSOC).


## Bugs and Colab

If you have any questions regarding this code or want to contribute mail me at:

patrick.korus@stud.tu-darmstadt.de
