#include "algorithm.h"
#pragma once

class OrigCTFAlgorithm : public CTFAlgorithm {
    vector<int> rowSDims;
    vector<int> colSDims;
    int setup_block_grid(vector<int>& row_dims, vector<int>& col_dims);
    void assign_initial_superpixels(map<int, vector<Point>>& superpixel_assigments);
    MoveSet get_potential_moves(Block* block);
    optional<PotentialMove> best_move(MoveSet& moves);
    using CTFAlgorithm::CTFAlgorithm;
};