#include "algorithm.h"
#include "engine.h"
#include "block.h"
#include <iostream>
#include <unordered_map>

using namespace cv;
using namespace std;

SegmentationAlgorithm::SegmentationAlgorithm(Engine& engine) : engine(engine) {};


// int GreedyCTFAlgorithm::run_level() {
//     vector<PotentialMove> moves;
//     unordered_map<Point, int> block_lookup;

//     vector<Block*> bblocks = engine.get_boundary_blocks();

//     for(auto block: bblocks) {
//         num_iterations++;
//         auto potentials = get_potential_moves(block);
//         auto move = best_move(potentials);
//         if(move) {
//             moves.push_back(*move);
//         }
//         else {
//             moves.push_back(PotentialMove({.delta=INFINITY, .block=nullptr, .new_sp=nullptr}));
//         }
        
//     }

//     sort(moves.begin(), moves.end());
//     for (auto it = moves.begin(); it != moves.end();  ++it )
//     {
//         block_lookup[it->block->index] = it - moves.begin();
//     }
    
//     int n = 5000;
//     while(n) {
//         auto& best_move = moves[0];
//         auto old_sp = best_move.block->superpixel;
//         if (best_)
//         execute_move(best_move);
//     }

//     for(num_blocks: )



// }

int CTFAlgorithm::run_level() {
    vector<Block*> q1 = engine.get_boundary_blocks();
    vector<Block*> q2 = {};
    
    total_energy = get_initial_energy();
    Mat_<bool> queued;
    
    engine.is_boundary_block.copyTo(queued);
    
    int num_qs = 0;
    
    while(q1.size()) {
        for (auto b : q1) {
            queued(b->index) = false;
            
            num_iterations++;
            
            auto moves = get_potential_moves(b);
            
            auto move = best_move(moves);
            vector<Block*> v(4, nullptr);
            
            if (move) {
                //cout << "Moving " << move->block->index << " from " << move->block->superpixel->id << " to " << move->new_sp->id << endl;
                //cout << "(Delta: " << move->delta << "); " << move->delta / energy_evaluator.get(*(move->block->superpixel)) << " of cost";
                execute_move(*move);
                
                v = {nullptr, nullptr, nullptr, nullptr};
                b->get_differently_labeled_neighbours(v);
                for (auto b2: v) {
                    if (!b2) { continue; }
                    if (!queued(b2->index)) {
//                    if (b2->get_is_boundary()) {
                        q2.push_back(b2);
                    }
                }
            }
        }
        num_qs++;
        q1 = q2;
        q2 = {};
    }
    cout << "Moves: " << moves_log.size() - 1 << endl;
    cout << "Iterations: " << num_iterations << endl;
    cout << "Qs: "<< num_qs << endl << endl
    ;
    energy_evaluator.flush();
    return 0;
}

// int RandomCTFAlgorithm::run_level() {
//     Mat_<bool> queued;
//     engine.is_boundary_block.copyTo(queued);
    
//     map<Point, Block*> boundary_blocks;
//     for(auto isbb: queued) {
//         boundary_blocks.ins
//     }
    
//     // pick a random bblock
//     // pop it, and insert
//     // pick another random one
// }

Energy CTFAlgorithm::get_initial_energy() {
    Energy energy = {0,0};
    for(auto &sp: engine.superpixels) {
        energy += energy_evaluator.get(sp);
    }
    return energy;
}

MoveSet CTFAlgorithm::get_potential_moves(Block *block) {
    MoveSet s;
    if (label_change_violates_connectivity(block) or label_change_violates_minsize(block)) {
        return s;
    }
    auto v = vector<Block*>(4, nullptr);
    block->get_differently_labeled_neighbours(v);
    for (Block* b : v) {
        if (!b) { continue; }
        Energy delta = energy_evaluator.try_potential_change(block, *(b->superpixel));
        s.insert({.delta=delta, .block=block, .new_sp=b->superpixel});
    }
    return s;
}

inline bool CTFAlgorithm::label_change_violates_connectivity(Block* block) {
//    return false;
    assert(block->level == engine.level);
    bool res = utils::is_connected_optimized(engine.blocks, block, block->index.y - 1, block->index.x - 1, block->index.y + 2, block->index.x + 2);
    return !res;
}

inline bool CTFAlgorithm::label_change_violates_minsize(Block* block) {
    return false;
    bool res = block->superpixel->sums.area - block->sums.area < block->superpixel->initial_size / 4;
    return res;
}

optional<PotentialMove> CTFAlgorithm::best_move(MoveSet& moves) {
    if(moves.size()) {
        auto move = *(moves.begin());
        if (double(move.delta) < 0) {
            return optional<PotentialMove>(move);
        }
        return optional<PotentialMove>();
    }
    else{
        return optional<PotentialMove>();
    }
}

void CTFAlgorithm::execute_move(PotentialMove& move) {
    energy_evaluator.flush(*move.block->superpixel);
    energy_evaluator.flush(*move.new_sp);
    engine.relabel(move.block, *(move.new_sp));
    total_energy += move.delta;
    Move log = {
        .energy=total_energy,
        .superpixel_id=move.new_sp->id,
        .block_index=move.block->index,
        .num_iterations=num_iterations,
        .level=engine.level,
    };
    moves_log.push_back(log);
}

Energy EnergyEvaluator::get(Superpixel& superpixel) {
    try {
        return sp_cache.at(superpixel.id);
    }
    catch (const std::out_of_range& ex) {
        Energy res = calculate(superpixel.sums);
        sp_cache[superpixel.id] = res;
        return res;
    }
}

void EnergyEvaluator::flush(Superpixel& superpixel) {
    sp_cache.erase(superpixel.id);
}

void EnergyEvaluator::flush() {
    sp_cache.clear();
}

Energy EnergyEvaluator::try_potential_change(Block* block, Superpixel& new_sp) {
    Energy orig_s1 = get(*block->superpixel);
    Energy orig_s2 = get(new_sp);
    
    sums s1 = block->superpixel->sums;
    sums s2 = new_sp.sums;
    
    s1 -= block->sums;
    s2 += block->sums;
    
    Energy res1 = calculate(s1);
    Energy res2 = calculate(s2);
    
    return (res1 + res2) - (orig_s1 + orig_s2);
}

//double EnergyEvaluator::get(Block* block) {
//    try {
//        return block_cache.at(block);
//    }
//    catch (const std::out_of_range& ex) {
//        double res = calculate(block);
//        block_cache[block] = res;
//        return res;
//    }
//}

Energy CTFEnergyEvaluator::calculate(sums& s) {
    return {.col_part=e_col(s), .reg_part=0};//e_reg(s)};
}

double CTFEnergyEvaluator::e_reg(sums& s) {
    Vec2f mean = s.pos_sum / s.area;
    Vec2f mean2;
    pow(mean, 2, mean2);
    auto res = Vec2f(s.pos2_sum) - 2 * mean.mul(Vec2f(s.pos_sum)) + mean2 * s.area;
    return sum(res)[0]; // / (double(s.area*s.area));// / double(s.number_blocks));
}

double CTFEnergyEvaluator::e_col(sums& s) {
    Scalar mean = s.col_sum / s.area;
    Scalar mean2;
    pow(mean, 2, mean2);
    Scalar res = s.col2_sum - 2 * mean.mul(s.col_sum) + mean2 * s.area;
    return sum(res)[0]; // / (double(s.area*s.area));//; / s.number_blocks);
}