# Continuous learing: Reacher (version 1)

Here we have adopted Deep Deterministic Policy Gradient (DDPG) algoritm from Continuous Control Project. There are two important changes with respect to the other porject:

* we have addopted the code in a way that it can deal with multiple agents.
* we have added decay rate in the nose from 1 by 0.9999 each time nose function is accessed.

## Actor/Critic Architecture

### Actor

```
  (fc1): Linear(in_features=33, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
```

### Critic

```
  (fcs1): Linear(in_features=33, out_features=128, bias=True)
  (fc2): Linear(in_features=132, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
```

## Other parameter

##### replay buffer size
BUFFER_SIZE = int(1e5)  
##### minibatch size
BATCH_SIZE = 128   
##### discount factor
GAMMA = 0.99  
##### tau for soft update of target parameters
TAU = 1e-2     
##### learning rate of the actor 
LR_ACTOR = 1e-4   
##### learning rate of the critic
LR_CRITIC = 1e-3 
##### L2 weight decay
WEIGHT_DECAY = 0  
##### Noise's starting sigma parameter
NOISE_SIGMA = 1     
##### Noise's starting theta parameter
NOISE_THETA = 1 
##### Noise's decay rate that is applied to both: sigma and theta.
DECAY_RATE = 0.9999     

## Result

Evirionment was solved in 608 episodes (gettig more then 0.5 points on averege among 100 episodes).


![Alt text](https://github.com/sbartek/unity_tennis_with_reinforcement_learning/blob/master/score1.png?raw=true "Optional Title")

One can observe trained agent running: `TennisPlayTrainedAgents.ipynb` notebook and compare it with random agent `TennisPlayRandomAgents.ipynb`.

## Future improvements

We should investigate different configurations of NN. Iw particular we could try to add dropout.