from .energy import EnergyEvaluator, RegEnergyTerm, ColEnergyTerm, PWLColEnergyTerm
from .utils import region_is_connected
import numpy as np
from collections import deque, defaultdict, Iterator
from blist import sortedlist, sorteddict, sortedset

class MonocularEnergyEvaluator(EnergyEvaluator):
    terms = [
        (1.0, RegEnergyTerm),
        (1.0, ColEnergyTerm),
    ]

class PWLEnergyEvaluator(EnergyEvaluator):
    terms = [
        (1.0, PWLColEnergyTerm),
    ]

class Algorithm:
    def __init__(self, engine):
        self.engine = engine
        self.finished = False

    @property
    def blocks(self):
        return self.engine.blocks


class BlockQueue(Iterator):
    def __init__(self, boundary_blocks):
        self.is_boundary_block = boundary_blocks.astype(bool)
        self._q1 = sortedlist((b for b in boundary_blocks.flat if b is not None), key=self._key)
        self._q2 = sortedlist(key=self._key)
        self._l = len(self._q1)
        self._pos = 0

    def __next__(self):
        try:
            res = self._q1[self._pos]
            self._pos += 1
            self._l -= 1
            return res
        except IndexError:
            self._pos = 0
            self._q1 = self._q2
            self._l = len(self._q1)
            self._q2 = sortedlist(key=self._key)
            raise StopIteration

    def _key(self, block):
        return block.index

    def __len__(self):
        return self._l

    def add(self, item):
        self._q2.add(item)

class CTFAlgorithm(Algorithm):
    def __init__(self, engine):
        super().__init__(engine)
        self.level = self.engine.starting_level
        self.moves_log = defaultdict(list)
        self.energy_evaluator = MonocularEnergyEvaluator(self.engine)
        self.number_passes = 1

    def label_change_violates_minsize(self, block):
        if block.superpixel.number_pixels - block.number_pixels <= block.superpixel.initial_size // 4:
            return True
        return False

    def label_change_violates_connectivity(self, block):
        return not region_is_connected(block.connection_nhood)

    def best_move(self, possible_moves):
        best_move = None
        for move in possible_moves:
            if move['total_delta'] < 0 and (best_move is None or move['total_delta'] < best_move['total_delta']):
                best_move = move
        return best_move

    def relabel_block(self, block, label):
        sp = self.engine.get_superpixel_by_label(label)
        old = block.superpixel
        block.superpixel.remove_block(block)
        sp.add_block(block)
        block.superpixel = sp
        self.energy_evaluator.notify_relabel(block, old)

    def get_boundary_blocks_queue(self):
        return BlockQueue(self.engine.get_boundary_blocks())

    def run_level(self):
        '''Main part of algorithm.'''
        self.energy_evaluator.initialize_with_superpixels(self.engine.superpixels)
        # q2 = deque([])

        num_iterations = 0
        moves_made = 0
        max_iterations = 400000

        # last = q[-1] 
        # import ipdb; ipdb.set_trace()

        for pass_no in range(self.number_passes):
            # find boundary blocks
            q = self.get_boundary_blocks_queue()
            # print("Pixels in list", len(q))

            # print("Level {} pass {}".format(self.level, pass_no+1))
            num_sets = 0

            while len(q) and num_iterations < max_iterations:
                num_sets += 1
                # print("Set", num_sets)
                for block in q:
                    num_iterations += 1
                    moves_made += 1

                    q.is_boundary_block[block.index] = False

                    # print("Trying px", block.index, "...")

                    if self.label_change_violates_connectivity(block) or self.label_change_violates_minsize(block):
                        continue

                    else:
                        possible_moves = [self.try_label(block, spixel) for spixel in block.neighbouring_different_spixels]

                        move = self.best_move(possible_moves)
                        # state['best_move'] = next((k, v) for k, v in state['moves'].items() if v['new_label'] == move['new_label']) if move is not None else move
                        # self.queue_state(state)

                        if move is not None:
                            # print('Moving', block.index, 'from', block.superpixel.id, 'to', move['new_label'])
                            moves_made += 1
                            self.moves_log[self.level].append((num_iterations, list(self.energy_evaluator.total_energy())))
                            self.relabel_block(block, move['new_label'])
                            # state['relabel'] = (block.index, move['new_label'])
                            # state['add_boundary_blocks'] = []
                            q.add(block)
                            for other in block.differently_labelled_neighbours:
                                if not q.is_boundary_block[other.index]:
                                    q.is_boundary_block[other.index] = True
                                    q.add(other)
        # print('=============')
        # print('Num iterations', num_iterations)

    def run_iteration(self):
        self.run_level()
        if self.level <= 0:
            self.finished = True
        else:
            self.level -= 1

    def try_label(self, block, new_sp):
        # try relabelling a block to `new_sp`.
        # calculate energy delta
        delta_1 = self.energy_evaluator.calculate_delta(block.superpixel, [], [block])
        delta_2 = self.energy_evaluator.calculate_delta(new_sp, [block], [])
        # block_change_data = block.superpixel.try_remove_block(block), new_sp.try_add_block(block)
        # boundary_len_change_data = block.get_boundaries_with_superpixels(block.superpixel.id, new_sp.id)

        # print("Delta: ", np.sum(delta_1 + delta_2))

        return {'new_label': new_sp.id, 'delta': delta_1 + delta_2, 'total_delta': np.sum(delta_1 + delta_2)}

class WorstBlockFirst(CTFAlgorithm):
    def run_level(self):
        self.energy_evaluator.initialize_with_superpixels(self.engine.superpixels)
        self.blocks_set = sortedset(key=lambda b: -self.block_potential_changes[b]['total_delta'] if self.block_potential_changes[b] is not None else float('inf'))
        self.block_potential_changes = {}
        boundary_blocks = self.engine.get_boundary_blocks()
        for block in boundary_blocks.flat:
            if block is None:
                continue
            if self.label_change_violates_connectivity(block) or self.label_change_violates_minsize(block):
                continue
            possible_moves = [self.try_label(block, spixel) for spixel in block.neighbouring_different_spixels]
            self.block_potential_changes[block] = self.best_move(possible_moves)
            self.blocks_set.add(block)
        
        # move the best one...
        n = 5000
        while n and len(self.block_potential_changes):
            best_block = self.blocks_set[0]
            old = best_block.superpixel
            change = self.block_potential_changes[best_block]
            self.relabel_block(best_block, change['new_label'])
            new = best_block.superpixel
            
            # update the dicts etc as required.
            for block in best_block.eight_neighbourhood:
                if self.label_change_violates_connectivity(block) or self.label_change_violates_minsize(block):
                    self.block_potential_changes[block] = None

            for block in best_block.differently_labelled_neighbours:
                if self.block_potential_changes[block] is not None:
                    possible_moves = [self.try_label(block, spixel) for spixel in block.neighbouring_different_spixels]
                    self.block_potential_changes[block] = self.best_move(possible_moves)

            for sp in [old, new]:
                for block in sp.blocks:
                    if block in self.blocks_set:
                        self.blocks_set.remove(block)
                        print("Removed", block)
                        possible_moves = [self.try_label(block, spixel) for spixel in block.neighbouring_different_spixels]
                        self.block_potential_changes[block] = self.best_move(possible_moves)
                        self.blocks_set.add(block)

            # import ipdb; ipdb.set_trace()
            n -= 1






class MoveAtEndOfSet(CTFAlgorithm):
    def run_level(self):
        '''Main part of algorithm.'''
        self.energy_evaluator.initialize_with_superpixels(self.engine.superpixels)
        # q2 = deque([])

        num_iterations = 0
        moves_made = 0
        max_iterations = 400000

        # last = q[-1] 
        # import ipdb; ipdb.set_trace()

        for pass_no in range(self.number_passes):
            # find boundary blocks
            q = self.get_boundary_blocks_queue()

            print("Level {} pass {}".format(self.level, pass_no+1))
            num_sets = 0

            while len(q) and num_iterations < max_iterations:
                moves_queue = []
                num_sets += 1
                print("Set", num_sets)

                for block in q:
                    num_iterations += 1
                    moves_made += 1

                    q.is_boundary_block[block.index] = False

                    if self.label_change_violates_connectivity(block) or self.label_change_violates_minsize(block):
                        continue

                    else:
                        possible_moves = [self.try_label(block, spixel) for spixel in block.neighbouring_different_spixels]

                        move = self.best_move(possible_moves)
                        # state['best_move'] = next((k, v) for k, v in state['moves'].items() if v['new_label'] == move['new_label']) if move is not None else move
                        # self.queue_state(state)

                        if move is not None:
                            # print('Moving', block.index, 'from', block.superpixel.id, 'to', move['new_label'])
                            moves_queue.append((block, move['new_label']))
                            # state['relabel'] = (block.index, move['new_label'])
                            # state['add_boundary_blocks'] = []

                for block, new_label in moves_queue:
                    if self.label_change_violates_connectivity(block) or self.label_change_violates_minsize(block):
                        continue
                    q.add(block)
                    for other in block.differently_labelled_neighbours:
                        if not q.is_boundary_block[other.index]:
                            q.is_boundary_block[other.index] = True
                            q.add(other)
                    self.moves_log[self.level].append((num_iterations, list(self.energy_evaluator.total_energy())))
                    self.relabel_block(block, new_label)

        # print('=============')
        # print('Num iterations', num_iterations)

class WorstFirstBlockQueue(BlockQueue):
    def __init__(self, boundary_blocks, energy_evaluator):
        self.energy_evaluator = energy_evaluator
        super().__init__(boundary_blocks)

    def _key(self, block):
        return -self.energy_evaluator.get(block.superpixel)

class WorstFirstCTFAlgorithm(CTFAlgorithm):
    def __init__(self, engine):
        super().__init__(engine)

    def get_boundary_blocks_queue(self):
        return WorstFirstBlockQueue(self.engine.get_boundary_blocks(), energy_evaluator=self.energy_evaluator)

class PWLWorstFirst(CTFAlgorithm):
    def __init__(self, engine):
        super().__init__(engine)
        self.energy_evaluator = PWLEnergyEvaluator(self.engine)

class ParallelCTFAlgorithm(CTFAlgorithm):
    def split_into_chunks(self):
        pass

class SACTFAlgorithm(WorstFirstCTFAlgorithm):
    pass