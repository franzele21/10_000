import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
import random
import numpy as np
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
                unique, counts = np.unique(dices, return_counts=True)
                dice_result = dict(zip(unique, counts))
                
                # loosing case 
                print(dice_result[5] if 5 in dice_result else 0)
                if dice_result[1] == 0 and dice_result[5] == 0:
                    self.state = self.base_state.copy()
                else:   # winning case
                    self.state[1] += dice_result[1]*100 + dice_result[5]*50
        return None # il faut définir une fonction de retour
    def pprint_state(self):
        return {k:v.item() for k,v in zip(["Total points", "Round points", "Remaining dice number"], self.state)}
                
    def __repr__(self):
        return str(self.pprint_state())
    
if __name__ == "__main__":
    env = Env10000()
    env.reset()
    print(env)
    print(env.step(THROW_DICE))
