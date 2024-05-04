from abc import abstractmethod
from typing import List

import numpy as np
from rlgym_sim.utils.gamestates import GameState, PlayerData


class AbstractRCF(object):
    def extract_replays(self, replays):
        extracted = []

        for state in replays:
            if self.replay_matching_condition(state):
                extracted.append(state)

        return np.array(extracted)

    def replay_matching_condition(self, state: GameState) -> bool:
        return self.replay_matching_condition_one_player(state, state.players[0])

    @abstractmethod
    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData) -> bool:
        pass
