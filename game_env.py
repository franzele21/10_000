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
        self.base_state = np.array([0, 0, max_dice_nb], dtype=np.int32)
        self.state = self.base_state.copy()
        
        # simpler and easier if we want to try multiple models
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.action_space = spaces.Discrete(2)
    
    def _state_to_obs(self):
        """Flatten + normalize state to float32 array matching observation_space."""
        t = float(self.state[0]) / float(max(1, self.objectiv))
        r = float(self.state[1]) / float(max(1, self.objectiv))
        rem = float(self.state[2]) / float(max(1, self.max_dice_nb))
        return np.array([t, r, rem], dtype=np.float32)
    
    def reset(self, *, seed = None, options = None):
        self.state = self.base_state.copy()
        obs = self._state_to_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        prev_state = self.state.copy()

        # convert actions in case they come as numpy/int
        action = int(action)

        if action == KEEP_GAINS:
            # bank the round points
            self.state[0] += self.state[1]
            self.state[1] = 0
            self.state[2] = self.max_dice_nb

        elif action == THROW_DICE:
            # results of the dice throw
            dices = np.random.randint(1, 6, self.state[2])
            unique, counts = np.unique(dices, return_counts=True)
            dice_result = dict(zip(unique, counts))

            # losing case
            if not dice_result.get(1, 0) and not dice_result.get(5, 0):
                self.state = self.base_state.copy()
            else:   # winning case
                self.state[1] += dice_result.get(1, 0)*100 + dice_result.get(5, 0)*50
                self.state[2] -= dice_result.get(1, 0) + dice_result.get(5, 0)
        else:
            raise ValueError(f"Invalid action: {action}")

        reward = self.return_function(prev_state, action)

        # termination when objective reached or exceeded
        terminated = bool(self.state[0] >= self.objectiv)
        truncated = False
        info = {}

        return self._state_to_obs(), float(reward), terminated, truncated, info
    
    
    def return_function(self, previous_state, action):
        """
        previous_state: the state BEFORE the action (tuple/array: total, round, remaining)
        action: either KEEP_GAINS or THROW_DICE

        Returns a float reward.
        - KEEP_GAINS: reward == points banked into total (cur_total - prev_total)
        - THROW_DICE:
            * if the throw caused a loss (round points lost), reward = -lost_round_points
            * otherwise reward = incremental round points gained by the throw
        """
        prev_total, prev_round, prev_remain = previous_state
        cur_total, cur_round, cur_remain = self.state

        reward = 0.0

        if action == KEEP_GAINS:
            # reward the agent for banking round points into the total
            reward = float(cur_total - prev_total)

        elif action == THROW_DICE:
            # losing throw: round points reset and agent lost the accumulated round points
            # detect loss by seeing round dropped to 0 while previous round > 0
            if cur_round == 0 and prev_round > 0 and cur_total <= prev_total:
                reward = float(-prev_round)
            else:
                # reward the incremental round points from this throw
                reward = float(cur_round - prev_round)
        return reward
    
    def pprint_state(self):
        return {k:v.item() for k,v in zip(["Total points", "Round points", "Remaining dice number"], self.state)}
                
    def __repr__(self):
        return str(self.pprint_state())
    
if __name__ == "__main__":
    
    env = Env10000()
    obs = env.reset()
    terminated = False

    print("Manual play: 't' = throw, 'k' = keep, 'q' = quit")
    while not terminated:
        print("State:", env.pprint_state())
        cmd = input("action (t/k/q): ").strip().lower()
        if cmd == "q":
            break
        action = THROW_DICE if cmd == "t" else KEEP_GAINS
        obs, reward, terminated, truncated, info = env.step(action)
        print(f" -> reward: {reward}, terminated: {terminated}")
    if terminated:
        print("Final:", env.pprint_state())
