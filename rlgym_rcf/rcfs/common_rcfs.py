from rlgym_sim.utils.gamestates import GameState

from rlgym_rcf.rcfs.rcf_abstract import AbstractRCF
from rlgym_rcf.utils.rcf_utils import _is_on_wall, _is_on_ceiling


class OnWallRCF(AbstractRCF):
    def replay_matching_condition(self, state: GameState):
        return _is_on_wall(state.players[0])


class OnCeilingRCF(AbstractRCF):
    def replay_matching_condition(self, state: GameState):
        return _is_on_ceiling(state.players[0])


class TouchingBallRCF(AbstractRCF):
    def replay_matching_condition(self, state: GameState):
        return state.players[0].ball_touched
