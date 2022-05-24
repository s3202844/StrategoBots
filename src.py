import numpy as np

from Scorer import step_score
from Random import RandomSelection
from stratego_env import StrategoMultiAgentEnv, ObservationComponents, ObservationModes, GameVersions


if __name__ == '__main__':
    config = {
        'version': GameVersions.STANDARD,
        'random_player_assignment': True,
        'human_inits': True,
        'observation_mode': ObservationModes.PARTIALLY_OBSERVABLE,
    }

    env = StrategoMultiAgentEnv(env_config=config)

    number_of_games = 1
    wons = [0, 0]
    for _ in range(number_of_games):
        print("New Game Started")
        obs = env.reset()
        pre_obs = None
        while True:

            assert len(obs.keys()) == 1
            current_player = list(obs.keys())[0]
            assert current_player == 1 or current_player == -1

            if current_player == 1:
                step_score(pre_obs, obs, current_player)
                pre_obs = obs

            current_player_action = RandomSelection(
                current_player=current_player, obs_from_env=obs)

            obs, rew, done, info = env.step(
                action_dict={current_player: current_player_action})
            # print(f"Player {current_player} made move {current_player_action}")

            if done["__all__"]:
                print(
                    f"Game Finished, player 1 rew: {rew[1]}, player -1 rew: {rew[-1]}")
                if rew[1] == 1.0:
                    wons[0] += 1
                elif rew[-1] == 1.0:
                    wons[1] += 1
                break
            else:
                assert all(r == 0.0 for r in rew.values())
    print(wons)
