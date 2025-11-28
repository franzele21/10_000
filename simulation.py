import os
from stable_baselines3 import PPO
from game_env import Env10000, THROW_DICE, KEEP_GAINS

MODEL_PATH = "ppo_10000_env" 
N_EPISODES = 10
MAX_STEPS = 1000

def simulate(model_path=MODEL_PATH, n_episodes=N_EPISODES):
    # load model (no VecNormalize used here; if you used VecNormalize during training,
    # load it similarly and use the normalized env for inference)
    model = PPO.load(model_path)

    env = Env10000()
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False
        step = 0
        ep_reward = 0.0
        print(f"\n=== Episode {ep} ===")
        while not done and step < MAX_STEPS:
            # model.predict expects the same obs format used in training
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            step += 1
            act_name = "T" if action == THROW_DICE else "K"
            print(f"Step {step:03d} | action={act_name} | reward={reward:7.1f} | state={env.pprint_state()}")
            done = bool(terminated or truncated)
        print(f"Episode {ep} finished in {step} steps, total reward={ep_reward:.1f}, final state={env.pprint_state()}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH + ".zip") and not os.path.exists(MODEL_PATH):
        print(f"Model '{MODEL_PATH}' not found in current folder.")
    else:
        simulate()