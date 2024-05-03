import numpy as np
from rlgym_sim.utils.common_values import SUPERSONIC_THRESHOLD
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


class SupersonicRCF(AbstractRCF):
    def replay_matching_condition(self, state: GameState):
        return np.linalg.norm(state.players[0].car_data.linear_velocity) >= SUPERSONIC_THRESHOLD


class DistToBallRCF(AbstractRCF):
    def __init__(self, distance_threshold: int = 1_000, before_threshold: bool = True):
        self.distance_threshold = distance_threshold
        self.before_threshold = before_threshold

    def replay_matching_condition(self, state: GameState):
        dist_to_ball = np.linalg.norm(state.ball.position - state.players[0].car_data.position)
        if self.before_threshold:
            return dist_to_ball <= self.distance_threshold
        return dist_to_ball >= self.distance_threshold
