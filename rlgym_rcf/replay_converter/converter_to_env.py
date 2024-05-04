import math
import pickle

import numpy as np
from rlgym.utils import StateSetter
from rlgym.utils.state_setters import StateWrapper

from rlgym_sim.utils import StateSetter as GymSetter
from rlgym_sim.utils.state_setters import StateWrapper as GymWrapper

from rlgym_rcf.utils.rcf_utils import _set_cars, _set_ball


class RCFSetterSim(StateSetter):
    def __init__(self, replay_files):
        self.replay_files = replay_files
        self.states = np.array([])
        self.probs = np.array([])

    def _create_dummy_states(self):
        return [pickle.dumps([0] * 93) * 2]

    def load(self):
        self.states = np.array([])
        for replay_file in self.replay_files:
            if self.states.size == 0:
                self.states = np.load(replay_file)
            else:
                self.states = np.concatenate((self.states, np.load(replay_file)), axis=0)

    def generate_probabilities(self):
        return np.ones((self.states.shape[0])) / self.states.shape[0]

    def reset(self, state_wrapper: StateWrapper):
        if self.states.size == 0:
            self.states = np.array(self._create_dummy_states())
        self.probs = self.generate_probabilities()

        state = np.array(pickle.loads(np.random.choice(a=self.states, p=self.probs, size=1)[0]))

        _set_ball(state_wrapper, state)
        _set_cars(state_wrapper, state)


class RCFSetterGym(GymSetter):
    def __init__(self, replay_files):
        self.replay_files = replay_files
        self.states = np.array([])
        self.probs = np.array([])

    def _create_dummy_states(self):
        return [pickle.dumps([0] * 93) * 2]

    def load(self):
        self.states = np.array([])
        for replay_file in self.replay_files:
            if self.states.size == 0:
                self.states = np.load(replay_file)
            else:
                self.states = np.concatenate((self.states, np.load(replay_file)), axis=0)

    def generate_probabilities(self):
        return np.ones((self.states.shape[0])) / self.states.shape[0]

    def reset(self, state_wrapper: GymWrapper):
        if self.states.size == 0:
            self.states = np.array(self._create_dummy_states())
        self.probs = self.generate_probabilities()

        state = np.array(pickle.loads(np.random.choice(a=self.states, p=self.probs, size=1)[0]))

        _set_ball(state_wrapper, state)
        _set_cars(state_wrapper, state)
