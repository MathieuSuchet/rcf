from rlgym_sim.utils.gamestates import GameState, PlayerData

from rlgym_rcf.rcfs.multi_rcf import MultiRCF
from rlgym_rcf.rcfs.rcf_abstract import AbstractRCF


class WrapperRCF(AbstractRCF):
    def __init__(self, rcf: AbstractRCF):
        self.rcf = rcf

    def replay_matching_condition_one_player(self, state: GameState, player: PlayerData):
        return self.rcf.replay_matching_condition_one_player(state, player)


class AllPlayerRCF(WrapperRCF):

    def replay_matching_condition(self, state: GameState):
        return all([self.replay_matching_condition_one_player(state, p) for p in state.players])


class AnyPlayerRCF(WrapperRCF):

    def replay_matching_condition(self, state: GameState):
        return any([self.replay_matching_condition_one_player(state, p) for p in state.players])


class OnePlayerRCF(WrapperRCF):
    def __init__(self, rcf: AbstractRCF, player_id: int):
        super().__init__(rcf)
        self.player_id = player_id

    def replay_matching_condition(self, state: GameState):
        player = None
        for p in state.players:
            if p.car_id == self.player_id:
                player = p

        if player is None:
            return False

        return self.replay_matching_condition_one_player(state, player)