#include "origctf.h"

using namespace cv;
using namespace std;

class CTFIEnergyEvaluator : public CTFEnergyEvaluator {
    virtual Energy calculate(sums& s) {
        return {.col_part=e_col(s), .reg_part=0};
    }
    using CTFEnergyEvaluator::CTFEnergyEvaluator;
};

class CTFIAlgorithm : public OrigCTFAlgorithm {
    CTFIEnergyEvaluator energy_evaluator;
    int setup_block_grid(vector<int>& row_dims, vector<int>& col_dims);
    void assign_initial_superpixels(map<int, vector<Point>>& superpixel_assigments);
    MoveSet get_potential_moves(Block* block);
    optional<PotentialMove> best_move(MoveSet& moves);
    using OrigCTFAlgorithm::OrigCTFAlgorithm;
    vector<vector<Point>> assignments;
};