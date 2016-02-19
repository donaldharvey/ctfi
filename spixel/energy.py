from collections import defaultdict, OrderedDict
import numpy as np
from .block import positions as SUMS_POSITIONS

class EnergyEvaluator:
    def __init__(self, engine):
        self.engine = engine
        self.terms = tuple((w, t(engine)) for w, t in self.terms)
        self.cache = {}

    def initialize_with_superpixels(self, superpixels):
        self.cache = {}
        for _, term in self.terms:
            term.initialize_with_superpixels(superpixels)

        for px in self.engine.superpixels:
            self.get(px, recalc=True)
        
    def get(self, superpixel, recalc=False):
        if recalc or superpixel not in self.cache:
            res = self.calculate(superpixel)
            self.cache[superpixel] = res
        else:
            res = self.cache[superpixel]
        return res

    def calculate(self, superpixel):
        return np.array([weight*term.get(superpixel) for weight, term in self.terms])

    def total_energy(self):
        return np.sum(self.get(px) for px in self.engine.superpixels)

    def calculate_with_potential_change(self, superpixel, add_blocks, remove_blocks):
        return np.array([weight*term.calculate_with_potential_change(superpixel, add_blocks, remove_blocks) for weight, term in self.terms])

    def calculate_delta(self, superpixel, add_blocks, remove_blocks):
        return self.calculate_with_potential_change(superpixel, add_blocks, remove_blocks) - self.calculate(superpixel) 

    def notify_relabel(self, block, old_superpixel):
        try:
            del self.cache[old_superpixel]
        except KeyError:
            pass
        try:
            del self.cache[block.superpixel]
        except KeyError:
            pass
        for _, term in self.terms:
            term.notify_relabel(block, old_superpixel)



# class BlockDependence:
#     pass


# class BlockOperation:
#     def __init__(self, type, blah):
#         pass

# class BlockPosTracker:
#     def __init__(self, superpixels):



# class SuperpixelDependence:
#     def calculate(self, superpixels):
#         pass


# class ComputationResult:
#     def __init__(self, result, internal=None):
#         self.result = result
#         self.internal = internal


class EnergyTerm:
    def __init__(self, engine):
        self.engine = engine
        self.cache = {}
        # self.block_info = defaultdict(OrderedDict)

    def initialize_with_superpixels(self, superpixels):
        self.cache = {}
        for px in superpixels:
            self.get(px, recalc=True)

    def get(self, superpixel, recalc=False):
        if recalc or superpixel not in self.cache:
            res = self.calculate(superpixel)
            self.cache[superpixel] = res
        else:
            res = self.cache[superpixel]
        return res

    def notify_relabel(self, block, old_superpixel):
        try:
            del self.cache[old_superpixel]
        except KeyError:
            pass
        try:
            del self.cache[block.superpixel]
        except KeyError:
            pass

    def calculate(self, superpixel, recalc=False):
        raise NotImplementedError

    def calculate_for_block(self, block):
        raise NotImplementedError

    def calculate_with_potential_change(self, superpixel, add_blocks, remove_blocks):
        raise NotImplementedError

class SummedEnergyTerm(EnergyTerm):
    def calculate(self, superpixel):
        return self.run(superpixel.sums)

    def calculate_with_potential_change(self, superpixel, add_blocks, remove_blocks):
        sums = superpixel.sums.copy()

        for b in add_blocks:
            sums += b.sums

        for b in remove_blocks:
            sums -= b.sums

        return self.run(sums)

    def run(self, sums):
        return self.f(sums)

class NormedSummedEnergyTerm(SummedEnergyTerm):
    def run(self, sums):
        return self.f(sums) / self.engine.blocks.size

# class SuperpixelMean:
#     def __init__(self, block_info, sum_shape):
#         self.block_info = block_info
#         self.cache = {}
#         self.sum_shape = sum_shape

#     def calculate(self, superpixel, recalc=False):
#         res = self.cache.get(superpixel.id)
#         if res is None or recalc:
#             vals = list(self.block_info[superpixel.id].values())
#             s = np.sum(vals, axis=0) if vals else np.zeros(self.sum_shape) # FIXME!    
#             n = len(superpixel.blocks) or 1
#             self.cache[superpixel.id] = (s,n)
#             return s/n
#         else:
#             return res[0] / res[1]

#     def calculate_with_change(self, superpixel, add_blocks=None, remove_blocks=None):
#         # import ipdb; ipdb.set_trace()
#         add_blocks = [] if add_blocks is None else add_blocks
#         remove_blocks = [] if add_blocks is None else remove_blocks
#         len_delta = len(add_blocks) - len(remove_blocks)
#         s,n = self.cache[superpixel.id]
#         sum_add = np.sum(np.array([self.block_info[block.superpixel.id][block.index] for block in add_blocks]), axis=0)
#         sum_remove = np.sum(np.array([self.block_info[block.superpixel.id][block.index] for block in remove_blocks]), axis=0)
#         new_s = s + (sum_add - sum_remove)
#         new_n = n + len_delta
#         return new_s/new_n

def p(s, prop):
    return s[SUMS_POSITIONS[prop]]

class RegEnergyTerm(NormedSummedEnergyTerm):
    def f(self, sums):
        s = lambda x: p(sums, x)
        mu = s("pos_sum") / s("number_pixels")
        return np.sum(s("pos2_sum") - 2*s("pos_sum")*mu + mu**2 * s("number_pixels")) / (s("number_pixels") / s("number_blocks"))


class ColEnergyTerm(NormedSummedEnergyTerm):
    def f(self, sums):
        s = lambda x: p(sums, x)
        mu = s("col_sum") / s("number_pixels")
        return np.sum(s("col2_sum") - 2*s("col_sum")*mu + mu**2 * s("number_pixels")) / (s("number_pixels") / s("number_blocks"))

class PWLColEnergyTerm(SummedEnergyTerm):
    def get_abc(self, sums):
        s = lambda x: p(sums, x)

        return np.linalg.solve(np.array([
            [s("pos2_sum")[1],   s("pos_cross_sum"), s('pos_sum')[1]],
            [s("pos_cross_sum"), s("pos2_sum")[0],   s('pos_sum')[0]],
            [s("pos_sum")[1],    s("pos_sum")[0],    1]
        ]), [s("col_x_sum"), s("col_y_sum"), s("col_sum")])

    def f(self, sums):
        s = lambda x: p(sums, x)
        a, b, c = self.get_abc(sums)

        return np.sum(
            s("col2_sum") 
            - 2 * (a*s('col_x_sum') + b*s('col_y_sum') + c*s('col_sum'))
            + 2 * a * b * s("pos_cross_sum") + 2 * a * c * s("pos_sum")[1] + 2 * b * c * s("pos_sum")[0]
            + a**2 * s("pos2_sum")[1] + b**2 * s("pos2_sum")[0] + c**2
        ) / (s("number_pixels") / s("number_blocks") * self.engine.blocks.size)

    


# class ColEnergyTerm(EnergyTerm):
#     def __init__(self, engine):
#         super().__init__(engine)
#         self.mean = SuperpixelMean(self.block_info, sum_shape=(3,2))
#         self.sum_cache = {}

#     def get_dependencies(self, block):
#         return np.array((block.mean_col, block.mean_col**2))

#     def get_mean(self, superpixel, recalc=False):
#         return self.mean.calculate(superpixel, recalc)

#     def calculate(self, superpixel, recalc=False):
#         mean_col, mean_col2 = self.get_mean(superpixel, recalc)
#         res = self.sum_cache.get(superpixel.id)
#         if res is None or recalc:
#             self.sum_cache[superpixel.id] = res = np.sum(list(self.block_info[superpixel.id].values()), axis=0)
#         col, col2 = res
#         number_blocks = len(superpixel.blocks)
#         return np.sum(col2 - 2*col*mean_col + mean_col2 * number_blocks) / (superpixel.number_pixels / number_blocks)

#     def calculate_with_potential_change(self, superpixel, add_blocks, remove_blocks):
#         mean_col, mean_col2 = self.mean.calculate_with_change(superpixel, add_blocks, remove_blocks)
#         col_delta = np.sum((b.mean_col for b in add_blocks), axis=0) - np.sum((b.mean_col for b in remove_blocks), axis=0)
#         col2_delta = np.sum((b.mean_col**2 for b in add_blocks), axis=0) - np.sum((b.mean_col**2 for b in remove_blocks), axis=0)
#         col, col2 = self.sum_cache[superpixel.id]
#         new_n = len(superpixel.blocks) + len(add_blocks) - len(remove_blocks)
#         new_number_pixels = superpixel.number_pixels + sum(b.number_pixels for b in add_blocks) - sum(b.number_pixels for b in remove_blocks)
#         return np.sum(col2 + col2_delta - 2*(col+col_delta)*mean_col + mean_col**2 * new_n) / (new_number_pixels / new_n)

# class BlockSum:
#     def calculate(self, blocks):
#         results = {block: self.op(block) for block in blocks}
#         return ComputationResult(sum(r.result for r in results.values()), internal={'results': results})

#     def calculate_with_change(self, res, add_blocks, remove_blocks):
#         new_res = {block: self.op.calculate_with_change(r, add_blocks, remove_blocks) for block, r in res.internal['results'].items()}
#         sum_add = {block: self.op(block) for block in add_blocks}
#         sum_remove = {block: self.op(block) for block in remove_blocks}
#         new_res.update(sum_add)
#         for k in sum_remove:
#             del new_res[k]
#         return ComputationResult(sum(r[1] for r in new_res.values(), internal={'results': new_res})