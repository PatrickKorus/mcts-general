# Monte Carlo Tree Search for OpenAI gym framework

General Python implementation of Monte Carlo Tree Search for the use with Open AI Gym environments.

The MCTS Algorithm is based on the one from [muzero-general](https://github.com/PatrickKorus/muzero-general) which is 
forked from [here](https://github.com/werner-duvaud/muzero-general).

This code was part of my Bachelor Thesis:

#### [An Evaluation of MCTS Methods for Continuous Control Tasks](https://github.com/PatrickKorus/muzero-general) 

The source code of the experiments covered by the thesis can be found [here](https://github.com/PatrickKorus/MCTSOC).

## Dependencies

Python 3.8 is used. Dependencies are mainly numpy and gym. Simply run:

```shell script
pip install -r requirements.txt
```

## How to use

```python
# Step up system recursion limit
import sys 
sys.setrecursionlimit(10000)
# TODO: Explain how to use agent
# TODO: Link to Thesis PDF
```

Disclaimer: No tested for envs with scalar action space