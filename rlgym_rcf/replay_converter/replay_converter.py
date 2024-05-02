import pickle

import numpy as np
import rlgym_sim
import tqdm
from rlgym_sim.utils.state_setters import StateWrapper

from rlgym_rcf.rcfs.rcf_abstract import AbstractRCF
from state_setter import ReplayToState
import multiprocessing as mp


def _worker(proc_num, state_setter: ReplayToState, states, queue, rcf, lock: mp.Lock, ready_lock: mp.Lock, verbose: int):
    env = rlgym_sim.make(
        team_size=3,
        spawn_opponents=True,
        state_setter=state_setter
    )

    if verbose > 0:
        print(f"[{proc_num}] - Waiting for others")
    with ready_lock:
        if verbose > 0:
            print(f"[{proc_num}] Ready")

    if verbose > 0:
        print(f"[{proc_num}] Started")

    for i in range(len(states)):
        _, info = env.reset(return_info=True)

        if rcf.replay_matching_condition(info["state"]):
            with lock:
                queue.put(pickle.dumps(StateWrapper(game_state=info["state"]).format_state()))

    queue.put(None)


class ReplayConverter(object):
    def __init__(self, rcf: AbstractRCF):
        self.rcf = rcf

    def extract_states(self, replays, n_proc: int = 8, filepath: str = "", verbose: int = 0):

        state_setter = ReplayToState(replays)
        n_matching_states = 0
        matching_states = []

        ready_lock = mp.Lock()
        ready_lock.acquire()
        queue = mp.Queue()
        lock = mp.Lock()

        processes = [mp.Process(target=_worker, args=(
            i,
            state_setter,
            replays[i * (replays.shape[0] // n_proc):(i + 1) * (replays.shape[0] // n_proc)],
            queue,
            self.rcf,
            lock,
            ready_lock,
            verbose)
                                ) for i in range(n_proc)
                     ]

        for p in processes:
            p.start()

        ready_lock.release()

        progress = tqdm.tqdm(desc="Matching states")

        while item := queue.get():
            progress.update()
            n_matching_states += 1
            matching_states.append(item)

        progress.close()

        for p in processes:
            p.terminate()

        data = np.array(matching_states)
        if verbose > 0:
            print(f"Found {n_matching_states} matching states")
        if len(filepath) > 0:
            self.__class__.save(filepath, data)
        return data

    @staticmethod
    def save(filepath, replays):
        np.save(filepath, replays)
        print(f"Successfully saved replays in {filepath}")