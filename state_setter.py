import numpy as np
from rlgym_sim.utils import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper

from rlgym_rcf.utils.rcf_utils import _set_cars, _set_ball


class ReplayToState(StateSetter):

    def __init__(self, states):
        self.states = states
        self.counter = 0

    def reset(self, state_wrapper: StateWrapper):
        while np.any(np.isnan(self.states[self.counter])):
            # print("Replay contains NaNs, ignoring")
            self.counter += 1

        _set_ball(state_wrapper, self.states[self.counter])
        _set_cars(state_wrapper, self.states[self.counter])

        if self.counter == 43:
            print("Stop")

        while np.any(np.isnan(state_wrapper.format_state())):
            # print("State containing NaNs, ignoring")
            self.counter += 1

            _set_ball(state_wrapper, self.states[self.counter])
            _set_cars(state_wrapper, self.states[self.counter])

        self.counter += 1
