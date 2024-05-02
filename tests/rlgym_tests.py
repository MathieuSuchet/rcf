import rlgym
from rlgym.gamelaunch import LaunchPreference
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

from rlgym_rcf.replay_converter.converter_to_env import ConverterToEnv

if __name__ == "__main__":
    env = rlgym.make(
        spawn_opponents=True,
        team_size=3,
        terminal_conditions=[TimeoutCondition(int(2 / (8 / 120)))],
        reward_fn=DefaultReward(),
        state_setter=ConverterToEnv(["replays/states_wall"]),
        action_parser=DiscreteAction(),
        obs_builder=AdvancedObs(),
        game_speed=1,
        launch_preference=LaunchPreference.STEAM
    )

    _, _ = env.reset(return_info=True)
    while True:
        actions = env.action_space.sample()
        _, _, terminal, _ = env.step(actions)

        if terminal:
            _, _ = env.reset(return_info=True)
