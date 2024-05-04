import numpy as np
from rlgym_sim.utils.common_values import SUPERSONIC_THRESHOLD, BLUE_TEAM, ORANGE_TEAM
from rlgym_sim.utils.gamestates import GameState, PlayerData

from rlgym_rcf.rcfs.rcf_abstract import AbstractRCF
from rlgym_rcf.utils.rcf_utils import _is_on_wall, _is_on_ceiling


class OnWallRCF(AbstractRCF):
    """
    Gets states where player 1 is on the wall
    """

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        return _is_on_wall(player)


class OnCeilingRCF(AbstractRCF):
    """
    Gets states where player 1 is on the ceiling
    """

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        return _is_on_ceiling(player)


class TouchingBallRCF(AbstractRCF):
    """
    Gets state were player 1 hits the ball
    """

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        return player.ball_touched


class SupersonicRCF(AbstractRCF):
    """
    Gets state where player 1 is supersonic
    """

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        return np.linalg.norm(player.car_data.linear_velocity) >= SUPERSONIC_THRESHOLD


class BallSpeedRCF(AbstractRCF):
    """
    Gets state where ball is inferior or superior to threshold
    """

    def __init__(self, speed_threshold: float = 1_000., before_threshold: bool = True):
        """
        Gets state where ball is inferior or superior to threshold
        :param speed_threshold: Ball speed which triggers the condition
        :param before_threshold: True if states have to be less than threshold, False otherwise
        """
        self.speed_threshold = speed_threshold
        self.before_threshold = before_threshold

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        ball_speed = np.linalg.norm(state.ball.linear_velocity)
        if self.before_threshold:
            return ball_speed <= self.speed_threshold
        return ball_speed >= self.speed_threshold


class DistToBallRCF(AbstractRCF):
    """
    Gets state where player 1's distance to ball is inferior or superior to threshold
    """

    def __init__(self, distance_threshold: float = 1_000., before_threshold: bool = True):
        """
        Gets state where player 1's distance to ball is inferior or superior to threshold
        :param distance_threshold: Player distance to ball which triggers the condition
        :param before_threshold: True if states have to be less than threshold, False otherwise
        """
        self.distance_threshold = distance_threshold
        self.before_threshold = before_threshold

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        dist_to_ball = np.linalg.norm(state.ball.position - player.car_data.position)
        if self.before_threshold:
            return dist_to_ball <= self.distance_threshold
        return dist_to_ball >= self.distance_threshold


class NCarsRCF(AbstractRCF):
    """
    Gets states with exactly n_blue blue players and n_orange orange players
    """

    def __init__(self, n_blue, n_orange):
        """
        Gets states with exactly n_blue blue players and n_orange orange players
        :param n_blue: Expected number of blue players
        :param n_orange: Expected number of orange players
        """
        self.n_blue = n_blue
        self.n_orange = n_orange

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        if len(state.players) != self.n_blue + self.n_orange:
            return False

        n_blue = n_orange = 0
        for p in state.players:
            if p.team_num == BLUE_TEAM:
                n_blue += 1
            elif p.team_num == ORANGE_TEAM:
                n_orange += 1

        return n_blue == self.n_blue and n_orange == n_orange


class AirRCF(AbstractRCF):
    def __init__(self, min_height: float = 200.):
        self.min_height = min_height

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData) -> bool:
        return not player.on_ground and player.car_data.position[2] >= self.min_height


class WallToAirDribbleRCF(AbstractRCF):
    def __init__(self, ball_dist_thresh: float = 300., orientation_threshold: float = 0.7):
        self.orientation_threshold = orientation_threshold
        self.ball_dist_thresh = ball_dist_thresh

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData) -> bool:
        dist_to_ball = np.linalg.norm(state.ball.position - player.car_data.position)
        dist_to_ball_cond = (
                 dist_to_ball <= self.ball_dist_thresh
        )

        ball_dir_from_p = (state.ball.position - player.car_data.position) / dist_to_ball
        up_forward = player.car_data.up() + player.car_data.forward()
        orientation_cond = np.dot(ball_dir_from_p, up_forward) >= self.orientation_threshold

        return dist_to_ball_cond and orientation_cond and _is_on_wall(player)
