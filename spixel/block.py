import numpy as np
from collections import defaultdict, namedtuple
from .utils import FOURDELTAS, EIGHTDELTAS

positions = {
    'pos_sum': slice(0,2),
    'pos2_sum': slice(2,4),
    'pos_cross_sum': 4,
    'col_sum': slice(5,8),
    'col2_sum': slice(8,11),
    'col_x_sum': slice(11,14),
    'col_y_sum': slice(14,17),
    'number_pixels': 17,
    'number_blocks': 18,
}

# class BlockRelabelling:
#     def __init__(self, relabels):
#         self.relabels = relabels
#         # eg. (6, 1, 2), (6, 2, 3)

#     @property 
#     def add_blocks(self):
#         b = []
#         for l in relabels:
#             return (l[[1], l[0], l[2]])


# block delta: superpixel add block
#                         remove block

def sum_n(n):
    return (n**2 + n) / 2

def sum_i_to_n(i, n):
    return sum_n(n) - sum_n(i-1)

def sum_n2(n):
    return (2*n**3 + 3*n**2 + n) / 6

def sum_i2_to_n2(i, n):
    return sum_n2(n) - sum_n2(i-1)

def sum_xy_over_range(i1, n1, i2, n2):
    return sum_i_to_n(i1, n1 - 1) * sum_i_to_n(i2, n2 - 1)

def sum_range(summer, a, b):
    return np.array([
        (b[1] - a[1]) * summer(a[0], b[0] - 1),
        (b[0] - a[0]) * summer(a[1], b[1] - 1)
    ])

# def squared_sum_range(a, b):
#     return np.array([(b[i] - a[i]) * sum_i2_to_n2(a[i], b[i] - 1) for i in (0,1)])

# (∑∑ x, ∑∑ y) = (n_y ∑x, n_x ∑y)
class Block:
    def __init__(self, pos, index, size, engine, level, sp=None):
        self.pos = pos
        self.size = size
        self.index = tuple(index)
        self.superpixel = sp
        self.engine = engine
        self._mean_col = None
        self.mean_pos = self.pos + (self.size / 2)
        self.level = level

        # ∑r
        pos_sum = sum_range(sum_i_to_n, self.pos, self.pos + self.size)

        # ∑r^2
        pos2_sum = sum_range(sum_i2_to_n2, self.pos, self.pos + self.size)

        # ∑xy
        pos_cross_sum = sum_xy_over_range(self.pos[0], self.pos[0]+self.size[0], self.pos[1], self.pos[1] + self.size[1])

        # ∑I(p)
        col_sum = np.sum(self.img_pixels, axis=(0,1))

        # ∑I(p)^2
        col2_sum = np.sum(self.img_pixels**2, axis=(0,1))

        # ∑I(p)y
        col_y_sum = np.sum(self.img_pixels * np.arange(self.pos[0], self.pos[0] + self.size[0])[:, np.newaxis, np.newaxis], axis=(0, 1))

        # ∑I(p)x
        col_x_sum = np.sum(self.img_pixels * np.arange(self.pos[1], self.pos[1] + self.size[1])[np.newaxis, :, np.newaxis], axis=(0, 1))

        self.sums = np.zeros(19, dtype='double')
        self.sums[positions['pos_sum']] = pos_sum
        self.sums[positions['pos2_sum']] = pos2_sum
        self.sums[positions['pos_cross_sum']] = pos_cross_sum
        self.sums[positions['col_sum']] = col_sum
        self.sums[positions['col2_sum']] = col2_sum
        self.sums[positions['number_pixels']] = self.number_pixels
        self.sums[positions['number_blocks']] = 1
        self.sums[positions['col_x_sum']] = col_x_sum
        self.sums[positions['col_y_sum']] = col_y_sum

        self._vfunc = np.vectorize(lambda a: a.superpixel.id == self.superpixel.id and a != self)

    def copy(self):
        return Block(pos=self.pos, index=self.index, size=self.size, engine=self.engine, sp=self.superpixel, level=self.level)

    def split(self, new_array_view, new_idx):
        if new_array_view.shape == (1,1):
            # straight copy.
            new_block = self.copy()
            new_block.index = tuple(new_idx)
            new_block.level -= 1
            new_array_view[0] = new_block
        
        # 00022
        # 00022
        # 00022
        # 11133
        # 11133

        
        elif new_array_view.shape == (1,2):
            new_size_0 = self.size // np.array([1, 2])
            new_size_1 = self.size - (new_size_0 * np.array([0, 1])) 
            new_array_view[0,0] = Block(pos=self.pos, index=new_idx, size=new_size_0, engine=self.engine, sp=self.superpixel, level=self.level - 1)
            new_array_view[0,1] = Block(pos=self.pos+(new_size_0*np.array([0,1])), index=new_idx+(0,1), size=new_size_1, engine=self.engine, sp=self.superpixel, level=self.level - 1)

        elif new_array_view.shape == (2,1):
            new_size_0 = self.size // np.array([2, 1])
            new_size_1 = self.size - (new_size_0 * np.array([1, 0])) 
            new_array_view[0,0] = Block(pos=self.pos, index=new_idx, size=new_size_0, engine=self.engine, sp=self.superpixel, level=self.level - 1)
            new_array_view[1,0] = Block(pos=self.pos+new_size_0*np.array([1,0]), index=new_idx+(1,0), size=new_size_1, engine=self.engine, sp=self.superpixel, level=self.level - 1)

        elif new_array_view.shape == (2,2):
            new_size_00 = self.size // np.array([2, 2])
            new_size_10 = np.array([self.size[0] - new_size_00[0], new_size_00[1]])
            new_size_01 = np.array([new_size_00[0], self.size[1] - new_size_00[1]])
            new_size_11 = self.size - (new_size_00)
            new_array_view[0,0] = Block(pos=self.pos, index=new_idx, size=new_size_00, engine=self.engine, sp=self.superpixel, level=self.level - 1)
            new_array_view[1,0] = Block(pos=self.pos+(new_size_00[0], 0), index=new_idx+(1,0), size=new_size_10, engine=self.engine, sp=self.superpixel, level=self.level - 1)
            new_array_view[0,1] = Block(pos=self.pos+(0, new_size_00[1]), index=new_idx+(0,1), size=new_size_01, engine=self.engine, sp=self.superpixel, level=self.level - 1)
            new_array_view[1,1] = Block(pos=self.pos+new_size_00, index=new_idx+(1,1), size=new_size_11, engine=self.engine, sp=self.superpixel, level=self.level - 1)
        
        self.superpixel.remove_block(self)
        for b in new_array_view.flat:
            b.superpixel.add_block(b, initial=True)

    def __eq__(self, other):
        return type(self) == type(other) and self.__key() == other.__key()

    def __key(self):
        return (tuple(self.pos), tuple(self.size), self.index, self.level)

    def __hash__(self):
        return hash(self.__key())

    def print_sums(self):
        return {k: self.sums[v] for k, v in positions}

    @property
    def four_neighbourhood(self):
        return self.get_four_neighbourhood(False, False)

    def get_four_neighbourhood(self, with_lengths=True, include_edges=True):
        # include_edges: e.g. image edges where one or two neighbours are None - return actual Nones.
        results = []
        for d in FOURDELTAS:
            res = self.engine.block_at(tuple(self.index + d)) 
            if with_lengths:
                l = self.size[0] if d[0] == 0 else self.size[1]
                if include_edges or res is not None:
                    results.append((l, res))
            else:
                if include_edges or res is not None:
                    results.append(res)
        return results

    @property 
    def img_indices(self):
        row, col = self.pos
        row_end, col_end = self.pos + self.size
        return np.s_[row:row_end, col:col_end]

    @property 
    def pixels(self):
        return set((y,x) for y in range(self.pos[0], self.pos[0] + self.size[0]) for x in range(self.pos[1], self.pos[1] + self.size[1]))

    @property 
    def height(self):
        return self.size[0]

    @property 
    def width(self):
        return self.size[1]

    @property
    def number_pixels(self):
        return np.product(self.size)

    @property 
    def img_pixels(self):
        return self.engine.img[self.img_indices]

    def get_boundary_length_with_other_block(self, b):
        if b not in self.four_neighbourhood:
            return 0
        else:
            if b.pos[0] == self.pos[0]:  # (same row)
                return b.size[0]
            else:
                return b.size[1]

    def get_boundaries_with_superpixels(self, *superpixel_labels):
        '''
        if superpixels provided, return a dict with {label: length} for each provided 
        spixel, and {None: all others}. 
        otherwise, use all superpixels as keys.'''
        lengths = defaultdict(int)
        for length, neighbour in self.get_four_neighbourhood(with_lengths=True, include_edges=True):
            l = None
            if neighbour is not None:
                if not superpixel_labels or neighbour.superpixel.id in superpixel_labels:
                    l = neighbour.superpixel.id

            lengths[l] += length
        return lengths


    @property
    def neighbouring_different_spixels(self):
        return {b.superpixel for b in self.four_neighbourhood if b.superpixel.id != self.superpixel.id}
    
    @property 
    def differently_labelled_neighbours(self):
        return [b for b in self.four_neighbourhood if b.superpixel.id != self.superpixel.id]

    @property 
    def eight_neighbourhood(self):
        results = []
        for d in EIGHTDELTAS:
            res = self.engine.block_at(tuple(self.index + d))
            if res is not None:
                results.append(res)
        return results

    @property 
    def connection_nhood(self):
        idx = np.array(self.index)
        reg = self.engine.blocks_region(start=idx-1, end=idx+2)
        return self._vfunc(reg)

    @property 
    def mean_col(self):
        if self._mean_col is None:
            self._mean_col = np.mean(self.img_pixels, axis=(0,1))
        return self._mean_col

    def __repr__(self):
        return 'Block(index={}, pos={}, size={}, superpixel={})'.format(self.index, self.pos, self.size, self.superpixel) 
