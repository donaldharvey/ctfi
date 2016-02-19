import numpy as np
from .utils import EIGHTDELTAS, bounded_at, ceil_div, partition_into_blocks, bounded_region, rgb_to_lab, grid_partition
from .superpixel import Superpixel
from .block import Block
from .algorithm import CTFAlgorithm
import operator
from itertools import chain, accumulate

from skimage.segmentation import mark_boundaries

class Engine:
    class defaults:
        number_superpixels = 100
        max_pixel_size = 16

    def __init__(self, img, algorithm_cls, params=None):
        if params is None:
            params = {}

        self.orig_img = img
        # self.img = self.orig_img.copy()
        self.img = rgb_to_lab(self.orig_img)
        self.params = {}
        self.params.update(self.defaults.__dict__)
        self.params.update(params)

        self.initialise_superpixels()
        self.populate_blocks()
        self.algorithm = algorithm_cls(self)
        

    def split_blocks(self, new_max_size):
        # import ipdb; ipdb.set_trace()
        max_size = 1
        row_sum = col_sum = 0

        for block in self.blocks[:,0]:
            row_sum += (1 if block.height == 1 else 2)
            if block.height > max_size:
                max_size = block.height

        for block in self.blocks[0,:]:
            col_sum += (1 if block.width == 1 else 2)
            if block.width > max_size:
                max_size = block.width

        

        if max_size == 1:
            return False

        blocks = np.empty((row_sum, col_sum), dtype=object)
        new_idx = np.array([0,0])

        new_row = 0

        # if self.algorithm.level == 0:
        #     import ipdb; ipdb.set_trace()

        for y in range(self.blocks.shape[0]):
            new_col = 0
            new_height = 1 if self.blocks[y][0].height == 1 else 2
            for x in range(self.blocks.shape[1]):
                block = self.blocks[y,x]
                new_width = (1 if block.width == 1 else 2)

                new_idx = np.array((new_row, new_col))
                end_idx = new_idx + (new_height, new_width)
                new_array_view = blocks[new_idx[0]:end_idx[0], new_idx[1]:end_idx[1]]

                block.split(new_array_view, new_idx)

                new_col += new_width
            new_row += new_height

        self.blocks = blocks

    def get_superpixel_by_label(self, label):
        return self.superpixels[label]

    def get_sp_boundary_img(self):
        boundaries = np.zeros(self.labels.shape, bool)
        for coords, label in np.ndenumerate(self.labels):
            nr_p = 0
            for d in EIGHTDELTAS:
                other = self.label_at(tuple(np.array(coords)+d))
                if other is not None and label != other and not boundaries[tuple(np.array(coords)+d)]:
                    nr_p += 1
            if nr_p >= 2:
                boundaries[coords] = True

        return boundaries

    def block_at(self, index):
        return bounded_at(self.blocks, index)

    def blocks_region(self, start, end):
        return bounded_region(self.blocks, start, end)

    def label_at(self, index):
        return bounded_at(self.labels, index)

    def run(self):
        while not self.algorithm.finished:
            self.algorithm.run_iteration()
#            import ipdb; ipdb.set_trace()
            if not self.algorithm.finished:
                self.split_blocks(self.max_block_size)
        # self.draw_boundaries()

    def get_results_dict(self):
        return {
            'seg': self.labels.tolist(),
            'moves_log': self.algorithm.moves_log,
        }

    # def queue_state(self, data):
    #     self.draw(data)
    #     print(self._last_updated)
    #     print(data)
    #     #import ipdb; ipdb.set_trace()

    # def draw(self, state_info):
    #     self._last_updated += 1
    #     _, isbb = self.get_boundary_blocks()

    #     fig, ax = plt.subplots()
    #     # im = self.img[:]
    #     # im[self.get_sp_boundary_img()] = [0,0,0]
    #     ax.imshow(self.img, interpolation='nearest')
    #     for coords, block in np.ndenumerate(self.blocks):
    #         coords = np.array(coords)
    #         fcol = 'sienna' if isbb[tuple(coords)] else 'none'
    #         cur = state_info.get('current_block')
    #         is_current = cur and cur.index == block.index
    #         if is_current:
    #             fcol = 'red'
    #         is_possible_move = block.index in state_info.get('moves', {})
    #         if is_possible_move:
    #             fcol = 'blue'

    #         r = plt.Rectangle([block.pos[1], block.pos[0]], block.size[1], block.size[0], facecolor=fcol, edgecolor='black')
    #         # check if is superpixel boundary...
    #         # for n in block.get_four_neighbourhood():
    #         #     if n.superpixel.id != block.superpixel.id:
    #         #         x1 = 

    #         ax.add_patch(r)

    #     plt.savefig("out/" + str(self._last_updated).rjust(6, '0') + ".png")

    def get_boundary_blocks(self):
        boundary_blocks = np.empty(self.blocks.shape, dtype=object)
        for block in self.blocks.flat:
            for other in block.four_neighbourhood:
                if other.superpixel != block.superpixel:
                    boundary_blocks[block.index] = block
                    break

        return boundary_blocks

    # def get_boundary_blocks(self):
    #     is_boundary_block = np.zeros(self.blocks.shape, dtype=bool)
    #     queue = [] 
    #     for block in self.blocks.flat:
    #         for other in block.four_neighbourhood:
    #             if other.superpixel != block.superpixel:
    #                 queue.append(block)
    #                 is_boundary_block[other.index] = True
    #                 break

    #     return queue, is_boundary_block

    def initialise_superpixels(self):
        '''Initialize a grid of superpixels with a grid size yielding a number of superpixels close to the requested number.'''
        self.labels = np.zeros(self.img.shape[:2], dtype=np.int32)
        self.superpixels = []

        grid_size = int(np.sqrt(self.labels.size / self.params['number_superpixels']))
        
        sp_rows, sp_cols = ceil_div(np.array(self.labels.shape), grid_size)
        grid_size

        for i in range(sp_rows):
            for j in range(sp_cols):
                y0, y1 = i*grid_size, (i+1)*grid_size
                x0, x1 = j*grid_size, (j+1)*grid_size
                y1 = min(self.labels.shape[0], y1)
                x1 = min(self.labels.shape[1], x1)
                # pixels = 
                # pixels = coords[:, i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
                px_id = len(self.superpixels)
                self.labels[y0:y1, x0:x1] = px_id
                self.superpixels.append(Superpixel((y1-y0)*(x1-x0), self, px_id))

        self.grid_size = grid_size

    # def initialize_superpixels(self):
    #     self.labels = np.zeros(self.img.shape[:2], dtype=np.int32)
    #     self.superpixels = []
    #     division = int(np.sqrt(self.params['number_superpixels']))
    #     num_rows, num_cols = grid_partition(self.labels.shape[0], division), grid_partition(self.labels.shape[1], division)
    #     for i in num_rows:  
    #         for j in num_cols:



    def get_block_dims(self):
        # calculate the dimensions of each block.

        # import ipdb; ipdb.set_trace()
        
        img_shape = self.img.shape[:2]
        block_dims = (row_dims, col_dims) = [np.zeros(s, dtype='int64') for s in self.blocks.shape]
        partitioned_dims = lambda a, b, n_a, n_b: np.array(([a] * n_a) + ([b] * n_b))

        # do the initial grid first then the remaining edge bits that don't cover a whole superpixel
        # a = big block width, n_a = number of big blocks, b = small block width, n_b = number of small blocks

        (a, b, n_a, n_b) = partition_into_blocks(self.grid_size, self.max_block_size)
        grid_covering = partitioned_dims(a, b, n_a, n_b)

        for i in (0, 1):  # first rows, then cols.
            res = np.array(np.tile(grid_covering, img_shape[i] // self.grid_size))
            block_dims[i][:len(res)] = res

            edge_width = img_shape[i] % self.grid_size
            if edge_width:
                (a, b, n_a, n_b) = partition_into_blocks(edge_width, self.max_block_size)
                block_dims[i][-(n_a + n_b):] = partitioned_dims(a, b, n_a, n_b) 

        return block_dims


    def populate_blocks(self):
        '''Set up the initial set of blocks.''' 

        # first, fix the base block geometry.

        init_division = max(2, ceil_div(self.grid_size, self.params['max_pixel_size']))
        self.max_block_size = ceil_div(self.grid_size, init_division) # largest block size, in pixels
        self.starting_level = int(np.ceil(np.log2(self.max_block_size)))

        blocks_shape = init_division * (np.array(self.labels.shape) // self.grid_size) + ceil_div(np.array(self.labels.shape) % self.grid_size, self.max_block_size)
        self.blocks = np.empty(blocks_shape, dtype=object)

        block_dims = self.get_block_dims()
        
        # now, with block_dims determined, initialise the block objects!

        for i, (block_height, y) in enumerate(zip(block_dims[0], accumulate(chain([0], block_dims[0]), operator.add))):
            for j, (block_width, x) in enumerate(zip(block_dims[1], accumulate(chain([0], block_dims[1]), operator.add))):
                sp = self.superpixels[self.labels[y, x]]

                b = Block(pos=np.array([y,x]), 
                    size=np.array([block_height, block_width]), index=(i,j), 
                    engine=self, level=self.starting_level)

                self.blocks[i, j] = b
                sp.add_block(b)

    def draw_boundaries(self):
        from matplotlib import pyplot as plt
        from PIL import Image

        plt.figure()

        other = np.array(Image.open('seg.png'))
        im = self.orig_img.copy()
        # for y in range(self.orig_img.shape[0]):
        #     for x in range(self.orig_img.shape[1]):
        #         a,b,c = self.algorithm.energy_evaluator.terms[1][1].get_abc(self.superpixels[self.labels[y,x]].sums)
        #         im[y,x] = a*x + b*y + c

        # # import ipdb; ipdb.set_trace()

        # im = color.lab2rgb(im)
        seg1 = mark_boundaries(im, self.labels, mode='subpixel', color=(1,0,0))
        seg2 = mark_boundaries(self.orig_img, other, mode='subpixel', color=(1,0,0))

        plt.imsave("our_seg.png", seg1)
        plt.imsave("cpp_seg.png", seg2)
        
        # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        energies = []
        for level, log in sorted(self.algorithm.moves_log.items(), reverse=True):
            energies += [x[1] for x in log]
            plt.axvline(len(energies))
        plt.stackplot(np.arange(len(energies)), np.array(energies).T)
        plt.show()
