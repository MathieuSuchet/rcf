from rlgym_sim.utils.gamestates import GameState

from rlgym_rcf.rcfs.rcf_abstract import AbstractRCF


class MultiRCF(AbstractRCF):
    def __init__(self, *rcfs: AbstractRCF):
        self.rcfs = rcfs

    def replay_matching_condition(self, state: GameState):
        for rcf in self.rcfs:
            if not rcf.replay_matching_condition(state):
                return False
        return True
