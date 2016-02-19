import numpy as np
from .block import positions

class Superpixel:
    def __init__(self, initial_size, engine, id):
        self.engine = engine
        self.id = id
        self.blocks = set()
        self.initial_size = initial_size
        self.sums = np.zeros(19, dtype='float64')
        self.number_blocks = 0
        # self.initialise_values()

    def __eq__(self, other):
        return type(self) == type(other) and self.__key() == other.__key()

    def __key(self):
        return self.id

    def __hash__(self):
        return hash(self.__key())

    def add_block(self, block, initial=False):
        '''Add a block to the superpixel and update all the metrics.'''
        if block.superpixel is not None and not initial:
            block.superpixel.remove_block(block)
        self.blocks.add(block)
        self.sums += block.sums
        self.number_blocks += 1
        block.superpixel = self
        self.engine.labels[block.img_indices] = self.id
        return block

    def print_sums(self):
        return {k: self.sums[v] for k, v in positions.items()}

    @property 
    def number_pixels(self):
        return self.sums[-2]

    def remove_block(self, block):
        self.blocks.remove(block)
        self.sums -= block.sums
        self.number_blocks -= 1
        block.superpixel = None
        self.engine.labels[block.img_indices] = -1
        return block

    # def rgb_squared_sum(self):
    #     return np.squared_sum(self.img_pixels**2)

    def __repr__(self):
        return "Superpixel(id={}, number_blocks={})".format(self.id, len(self.blocks))
