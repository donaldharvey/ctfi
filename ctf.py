#!/usr/bin/env python3

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from spixel.engine import Engine
from spixel.algorithm import CTFAlgorithm, WorstFirstCTFAlgorithm, MoveAtEndOfSet, PWLWorstFirst
from shutil import copy
from pathlib import Path
import sh
import json
import sys
# def app_energy(rgb_sums, rgb_squared_sums, size, num_blocks):
#     mean_rgb = rgb_sums / size
#     dif_sq = (rgb_squared_sums) - 2 * mean_rgb * rgb_sums + size * mean_rgb**2
#     return np.sum(dif_sq) / (size/num_blocks)

# def reg_energy(sum_pos, sum_squared_pos, size, num_blocks):
#     mean_pos = sum_pos / size
#     dif_sq = sum_squared_pos - 2 * mean_pos * sum_pos + size * mean_pos**2
#     return np.sum(dif_sq) / (size/num_blocks)

# class EnergyDependence:
#     pass

ctf_cpp = sh.Command("/me/w/code/spixel/spixel")

# class CTFEnergyEvaluator:
if __name__ == '__main__':
    from matplotlib.image import imread
    # f = '/me/w/proj/slic/SLIC-Superpixels/BSDS500/lion/lion 100x100.jpeg'
    # copy(f, str(Path(__file__).parent))
    # loc = Path(__file__).parent / Path(f).name
    # print(ctf_cpp("/me/w/code/spixel/examples/basic/config_lion.yml", loc))
    # (Path('seg') / (Path(f).stem + '.png')).rename(Path('seg.png'))
    e = Engine(imread(sys.argv[1]), CTFAlgorithm, {'number_superpixels': 100, 'max_pixel_size': 16})
    e.run()
    json.dump(e.get_results_dict(), sys.stdout)

