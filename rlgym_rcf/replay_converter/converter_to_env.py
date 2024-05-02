import math
import pickle

import numpy as np
from rlgym_sim.utils import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper


class ConverterToEnv(StateSetter):
    def __init__(self, replay_files):
        self.replay_files = replay_files
        self.states = np.array([])
        self.probs = np.array([])

    def load(self):
        self.states = np.array([])
        for replay_file in self.replay_files:
            if self.states.size == 0:
                self.states = np.load(replay_file)
            else:
                self.states = np.concatenate(self.states, np.load(replay_file))

    def generate_probabilities(self):
        return np.ones((self.states.shape[0])) / self.states.shape[0]

    def reset(self, state_wrapper: StateWrapper):
        self.probs = self.generate_probabilities()
        state = pickle.loads(np.random.choice(a=self.states, p=self.probs, size=1)[0])
        print(np.array(state[19:22]) * (180 / math.pi))
