import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
import random
import numpy as np
from stable_baselines3.common.env_checker import check_env
DICE_NB = 5
THROW_DICE = 0
KEEP_GAINS = 1

class Env10000(Env):
    def __init__(self, max_dice_nb=DICE_NB, point_objectiv=1000):
        super().__init__()
        self.max_dice_nb = max_dice_nb
        self.objectiv = point_objectiv

        # total points, round points, nb of remaining dices
        self.base_state = np.array([0, 0, max_dice_nb])
        self.state = self.base_state.copy()

        self.action_space = spaces.Discrete(2)
    
    def reset(self, *, seed = None, return_info = False, options = None):
        self.state = self.base_state.copy()
        
        return self.state
    
    def step(self, action):
        match action:
            case 1:
                self.state[0] += self.state[1]
                self.state[1] = 0
                self.state[2] = self.max_dice_nb
            case 0:
                # results of the dice throw
                dices = np.random.randint(1, 6, self.state[2])
                dices = list(map(lambda x: max(x%2, x%3, x), dices))
                print(dices)

    def pprint_state(self):
        return {k:v.item() for k,v in zip(["Total points", "Round points", "Remaining dice number"], self.state)}
                
    def __repr__(self):
        return str(self.pprint_state())