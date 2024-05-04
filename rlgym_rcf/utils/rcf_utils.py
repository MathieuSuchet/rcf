import numpy as np
from rlgym_sim.utils.common_values import SIDE_WALL_X, BACK_WALL_Y, CEILING_Z
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition

X_AT_ZERO = 8064


def _set_cars(state_wrapper: StateWrapper, data: np.ndarray):
    """
    Sets the players according to the game state from replay

    :param state_wrapper: StateWrapper object to be modified with desired state values.
    :param data: Numpy array from the replay to get values from.
    """

    data = np.split(data[9:], len(state_wrapper.cars))
    if len(data[0]) == 14:
        for i, car in enumerate(state_wrapper.cars):
            car.set_pos(*data[i][1:4])
            car.set_lin_vel(*data[i][4:7])
            car.set_ang_vel(*data[i][7:10])
            car.set_rot(*data[i][10:13])
            car.boost = data[i][13]
    else:
        for i, car in enumerate(state_wrapper.cars):
            car.set_pos(*data[i][:3])
            car.set_rot(*data[i][3:6])
            car.set_lin_vel(*data[i][6:9])
            car.set_ang_vel(*data[i][9:12])
            car.boost = data[i][12]


def _set_ball(state_wrapper: StateWrapper, data: np.ndarray):
    """
    Sets the ball according to the game state from replay

    :param state_wrapper: StateWrapper object to be modified with desired state values.
    :param data: Numpy array from the replay to get values from.
    """
    state_wrapper.ball.set_pos(*data[:3])
    state_wrapper.ball.set_lin_vel(*data[3:6])
    state_wrapper.ball.set_ang_vel(*data[6:9])


def _is_goal_scored(state: GameState):
    return GoalScoredCondition().is_terminal(state)


def _is_on_wall(car: PlayerData, wall_width_tolerance: float = 100., wall_height_tolerance: float = 100.) -> bool:
    """
        Detects whether the car is on the wall
        :param car: The car to compare
        :param wall_width_tolerance: Wall width tolerance (too low and it might under-trigger, too high and it may over-trigger)
        :param wall_height_tolerance: Wall height tolerance, distance from ground and off ceiling, you are on the wall when ground + wall_height_tolerance < height < CEILING_Z - wall_height_tolerance
        :return: True if on wall, False otherwise
    """
    on_flat_wall = (
            car.on_ground
            # Side wall comparison
            and (SIDE_WALL_X - wall_width_tolerance
                 < abs(car.car_data.position[0])
                 < SIDE_WALL_X + wall_width_tolerance
                 # Back wall comparison
                 or BACK_WALL_Y - wall_width_tolerance
                 < abs(car.car_data.position[1])
                 < BACK_WALL_Y + wall_width_tolerance)
            # Ceiling/Ground comparison
            and wall_height_tolerance
            < car.car_data.position[2]
            < CEILING_Z - wall_height_tolerance
    )

    if on_flat_wall:
        return True

    is_on_corner = False

    for a in (-1, 1):
        if is_on_corner:
            break

        for b in (-1, 1):
            if (car.car_data.position[1] - wall_width_tolerance
                    < a * car.car_data.position[0] + (X_AT_ZERO * b)
                    < car.car_data.position[1] + wall_width_tolerance):
                # On wall
                is_on_corner = True
                break

    return is_on_corner


def _is_on_ceiling(car: PlayerData, wall_width_tolerance: float = 100., wall_height_tolerance: float = 100.) -> bool:
    """
    Detects whether the car is on the ceiling
    :param car: The car to compare
    :param wall_width_tolerance: Wall width tolerance (too low and it might under-trigger, too high and it may over-trigger)
    :param wall_height_tolerance: Wall height tolerance, distance off ceiling before being considered on ceiling
    :return: True if on ceiling, False otherwise
    """
    return (
            car.on_ground
            and abs(car.car_data.position[0]) < SIDE_WALL_X - wall_width_tolerance
            and abs(car.car_data.position[1]) < BACK_WALL_Y - wall_width_tolerance
            and car.car_data.position[2] > CEILING_Z - wall_height_tolerance
    )
