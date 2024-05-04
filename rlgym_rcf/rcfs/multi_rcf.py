from abc import ABC

from rlgym_sim.utils.gamestates import GameState, PlayerData

from rlgym_rcf.rcfs.rcf_abstract import AbstractRCF


class MultiRCF(AbstractRCF, ABC):

    def __init__(self, *rcfs: AbstractRCF):
        self.rcfs = rcfs


class AllRCF(MultiRCF):
    def replay_matching_condition(self, state: GameState):
        return all([rcf.replay_matching_condition(state) for rcf in self.rcfs])

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        return all([rcf.replay_matching_condition_one_player(state, player) for rcf in self.rcfs])


class AnyRCF(MultiRCF):
    def replay_matching_condition(self, state: GameState):
        return any([rcf.replay_matching_condition(state) for rcf in self.rcfs])

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        return any([rcf.replay_matching_condition_one_player(state, player) for rcf in self.rcfs])


class NotRCF(MultiRCF):
    def replay_matching_condition(self, state: GameState):
        return not any([rcf.replay_matching_condition(state) for rcf in self.rcfs])

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        return not any([rcf.replay_matching_condition_one_player(state, player) for rcf in self.rcfs])
