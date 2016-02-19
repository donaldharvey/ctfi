import numpy as np

FOURDELTAS = np.array([(-1,0),(1,0),(0,1),(0,-1)])
EIGHTDELTAS = np.array(list(FOURDELTAS) + [(-1,-1), (-1,1), (1,-1), (1,1)])

def ceil_div(a, b):
    return np.floor_divide(a + b - 1, b)

def grid_partition(number_items, length):
    '''Return the dimensions of `length` partitioned into a `number_items`-sized set of equally-sized (or as close as possible) lengths.'''
    n_each, rem = divmod(length, number_items)
    sizes = np.array([int(n_each)] * number_items)
    addition, rem = divmod(rem, number_items)
    sizes += int(addition)
    indices = (np.arange(1, rem+1) * number_items/rem - 1).astype(int)
    sizes[indices] += 1
    return sizes

def np_memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        __slots__ = ()

        def __call__(self, key):
            try:
                return self[tuple(key.flat)]
            except KeyError:
                ret = self[tuple(key.flat)] = f(key)
                return ret

    return memodict()


class UnionFind:
    def __init__(self, size):
        self.items = np.full(shape=size, fill_value=-1, dtype=int)
        
    def find(self, e):
        root = e
        while self.items[root] >= 0:
            root = self.items[root]
        if e != root:
            while self.items[e] != root:
                tmpe = self.items[e]
                self.items[e] = root
                e = tmpe
        return root
    
    def union(self, e1, e2):
        e1 = self.find(e1)
        e2 = self.find(e2)
        if e1 == e2:
            return
        size1 = -self.items[e1]
        size2 = -self.items[e2]
        if size1 < size2:
            self.items[e1] = e2
            self.items[e2] = -(size1 + size2)
        else:
            self.items[e2] = e1
            self.items[e1] = -(size1 + size2)
            
    def size(self, e):
        return -self.items[self.find(e)]


def partition_into_blocks(length, max_block_size): 
    '''
    Divide a length `grid_size` into a partition of blocks of max length `max_block_size`.
    Resulting blocks have either size `max_bs` or `min_bs`.
    Returns: (max_bs, min_bs, number_max_blocks, number_min_blocks)
    '''
    # how many blocks will there be?
    div = ceil_div(length, max_block_size) 

    # actual max block size and min block size
    max_bs = ceil_div(length, div)
    min_bs = length // div

    number_max_blocks = length % div
    number_min_blocks = div - number_max_blocks

    return (max_bs, min_bs, number_max_blocks, number_min_blocks)


def bounded_at(ar, index):
    if np.any(np.array(index) < 0) or np.any(np.array(index) >= ar.shape):
        return None

    return ar[index]

def bounded_region(ar, start, end):
    start[0] = max(start[0], 0)
    start[0] = min(start[0], ar.shape[0]-1)
    start[1] = max(start[1], 0)
    start[1] = min(start[1], ar.shape[1]-1)
    end[0] = max(end[0], 0)
    end[0] = min(end[0], ar.shape[0]-1)
    end[1] = max(end[1], 0)
    end[1] = min(end[1], ar.shape[1]-1)
    res = ar[start[0]:end[0], start[1]:end[1]]
    return res

@np_memoize
def region_is_connected(sq):
    comp = np.full(shape=(sq.shape[0]+1,sq.shape[1]+1), fill_value=-1)
    count = 0
    uf = UnionFind(sq.shape[0] * sq.shape[1])

    # import ipdb; ipdb.set_trace()

    for idx, p in np.ndenumerate(sq):
        coords = np.array(idx)
        if p:
            top_i = comp[tuple(coords + [0, 1])]
            left_i = comp[tuple(coords + [1, 0])]
            if top_i < 0:
                if left_i < 0:
                    comp[tuple(coords + [1, 1])] = count
                    count += 1
                else:
                    comp[tuple(coords + [1, 1])] = left_i
            else:
                if left_i < 0:
                    comp[tuple(coords + [1, 1])] = top_i
                else:
                    comp[tuple(coords + [1, 1])] = left_i
                    # import ipdb; ipdb.set_trace()
                    uf.union(left_i, top_i)
    
    return uf.size(0) == count

from bisect import bisect_left, bisect_right

class SortedCollection:
    '''Sequence sorted by a key function.

    SortedCollection() is much easier to work with than using bisect() directly.
    It supports key functions like those use in sorted(), min(), and max().
    The result of the key function call is saved so that keys can be searched
    efficiently.

    Instead of returning an insertion-point which can be hard to interpret, the
    five find-methods return a specific item in the sequence. They can scan for
    exact matches, the last item less-than-or-equal to a key, or the first item
    greater-than-or-equal to a key.

    Once found, an item's ordinal position can be located with the index() method.
    New items can be added with the insert() and insert_right() methods.
    Old items can be deleted with the remove() method.

    The usual sequence methods are provided to support indexing, slicing,
    length lookup, clearing, copying, forward and reverse iteration, contains
    checking, item counts, item removal, and a nice looking repr.

    Finding and indexing are O(log n) operations while iteration and insertion
    are O(n).  The initial sort is O(n log n).

    The key function is stored in the 'key' attibute for easy introspection or
    so that you can assign a new key function (triggering an automatic re-sort).

    In short, the class was designed to handle all of the common use cases for
    bisect but with a simpler API and support for key functions.

    >>> from pprint import pprint
    >>> from operator import itemgetter

    >>> s = SortedCollection(key=itemgetter(2))
    >>> for record in [
    ...         ('roger', 'young', 30),
    ...         ('angela', 'jones', 28),
    ...         ('bill', 'smith', 22),
    ...         ('david', 'thomas', 32)]:
    ...     s.insert(record)

    >>> pprint(list(s))         # show records sorted by age
    [('bill', 'smith', 22),
     ('angela', 'jones', 28),
     ('roger', 'young', 30),
     ('david', 'thomas', 32)]

    >>> s.find_le(29)           # find oldest person aged 29 or younger
    ('angela', 'jones', 28)
    >>> s.find_lt(28)           # find oldest person under 28
    ('bill', 'smith', 22)
    >>> s.find_gt(28)           # find youngest person over 28
    ('roger', 'young', 30)

    >>> r = s.find_ge(32)       # find youngest person aged 32 or older
    >>> s.index(r)              # get the index of their record
    3
    >>> s[3]                    # fetch the record at that index
    ('david', 'thomas', 32)

    >>> s.key = itemgetter(0)   # now sort by first name
    >>> pprint(list(s))
    [('angela', 'jones', 28),
     ('bill', 'smith', 22),
     ('david', 'thomas', 32),
     ('roger', 'young', 30)]

    '''

    import collections

    def __init__(self, iterable=(), key=None):
        self._given_key = key
        key = (lambda x: x) if key is None else key
        decorated = sorted((key(item), item) for item in iterable)
        self._keys = collections.deque([k for k, item in decorated])
        self._items = collections.deque([item for k, item in decorated])
        self._key = key

    def _getkey(self):
        return self._key

    def _setkey(self, key):
        if key is not self._key:
            self.__init__(self._items, key=key)

    def _delkey(self):
        self._setkey(None)

    key = property(_getkey, _setkey, _delkey, 'key function')

    def clear(self):
        self.__init__([], self._key)

    def copy(self):
        return self.__class__(self, self._key)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def __reversed__(self):
        return reversed(self._items)

    def __repr__(self):
        return '%s(%r, key=%s)' % (
            self.__class__.__name__,
            self._items,
            getattr(self._given_key, '__name__', repr(self._given_key))
        )

    def __reduce__(self):
        return self.__class__, (self._items, self._given_key)

    def __contains__(self, item):
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return item in self._items[i:j]

    def index(self, item):
        'Find the position of an item.  Raise ValueError if not found.'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return self._items[i:j].index(item) + i

    def count(self, item):
        'Return number of occurrences of item'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return self._items[i:j].count(item)

    def insert(self, item):
        'Insert a new item.  If equal keys are found, add to the left'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    def insert_right(self, item):
        'Insert a new item.  If equal keys are found, add to the right'
        k = self._key(item)
        i = bisect_right(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    def popleft(self):
        self._keys.append()

    def remove(self, item):
        'Remove first occurence of item.  Raise ValueError if not found'
        i = self.index(item)
        del self._keys[i]
        del self._items[i]

    def find(self, k):
        'Return first item with a key == k.  Raise ValueError if not found.'
        i = bisect_left(self._keys, k)
        if i != len(self) and self._keys[i] == k:
            return self._items[i]
        raise ValueError('No item found with key equal to: %r' % (k,))

    def find_le(self, k):
        'Return last item with a key <= k.  Raise ValueError if not found.'
        i = bisect_right(self._keys, k)
        if i:
            return self._items[i-1]
        raise ValueError('No item found with key at or below: %r' % (k,))

    def find_lt(self, k):
        'Return last item with a key < k.  Raise ValueError if not found.'
        i = bisect_left(self._keys, k)
        if i:
            return self._items[i-1]
        raise ValueError('No item found with key below: %r' % (k,))

    def find_ge(self, k):
        'Return first item with a key >= equal to k.  Raise ValueError if not found'
        i = bisect_left(self._keys, k)
        if i != len(self):
            return self._items[i]
        raise ValueError('No item found with key at or above: %r' % (k,))

    def find_gt(self, k):
        'Return first item with a key > k.  Raise ValueError if not found'
        i = bisect_right(self._keys, k)
        if i != len(self):
            return self._items[i]
        raise ValueError('No item found with key above: %r' % (k,))

def rgb_to_lab(img):
    # const int RGB2LABCONVERTER_XYZ_TABLE_SIZE = 1024;
    table_size = 1024
    # // CIE standard parameters
    epsilon = 0.008856
    kappa = 903.3
    
    reference_white = (0.950456, 1.0, 1.088754)
    
    # Maximum values
    max_xyz_values = (0.95047, 1.0, 1.08883)

    sRGB_gamma_corrections = np.zeros(256)

    for pixel_value in range(256):
        normalized_value = pixel_value / 255
        transformed_value = normalized_value / 12.92 if normalized_value <= 0.04045 else ((normalized_value+0.055) / 1.055)**2.4
        sRGB_gamma_corrections[pixel_value] = transformed_value

    xyz_table_index_coefs = (table_size - 1) / np.array(max_xyz_values)

    f_xyz_conversions = np.zeros((3,table_size))
    for xyz_index, xyz_value in enumerate(max_xyz_values):
        step_value = xyz_value / table_size
        for i in range(table_size):
            orig = step_value*i
            normalized = orig / reference_white[xyz_index]
            transformed = normalized**(1/3) if normalized > epsilon else (kappa*normalized + 16) / 116

            f_xyz_conversions[xyz_index][i] = transformed

    def lab(px):
        corrected = np.array([sRGB_gamma_corrections[i] for i in reversed(px)])
        to_xyz = np.array([[
            [0.4124564, 0.357576, 0.1804375],
            [0.2126729, 0.715152, 0.0721750],
            [0.0193339, 0.119192, 0.9503041],
        ]])
        xyz = to_xyz.dot(corrected)[0]
        table_index = (xyz * xyz_table_index_coefs + 0.5).astype(int)
        f = np.array([f_xyz_conversions[x][idx] for x, idx in enumerate(table_index)])

        return [116 * f[1] - 16, 500*(f[0] - f[1]), 200*(f[1]-f[2])]

    return np.apply_along_axis(lab, 2, img)