//
//  algorithm.h
//  ctfi
//
//  Created by Donald Harvey on 27/02/2016.
//  Copyright © 2016 Donald S. F. Harvey. All rights reserved.
//

#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "superpixel.h"
#include "block.h"
#include "utils.h"
#include <boost/optional/optional.hpp>
#include <unordered_map>
#include <set>

using namespace cv;
using namespace std;

template<typename T>
using optional = boost::optional<T>;

struct Energy
{
    double col_part;
    double reg_part;
    Energy operator+(const Energy& e) {
        Energy e2 = {.col_part=col_part + e.col_part, .reg_part=reg_part + e.reg_part};
        return e2;
    }
    bool operator<(const Energy& other) const {
        return col_part + reg_part < other.col_part + other.reg_part;
    }
    bool operator>(const Energy& other) const {
        return col_part + reg_part > other.col_part + other.reg_part;
    }
    bool operator<(const double& other) const {
        return col_part + reg_part < other;
    }
    bool operator>(const double& other) const {
        return col_part + reg_part < other;
    }
    void operator+=(const Energy& other) {
        col_part += other.col_part;
        reg_part += other.reg_part;
    }
    operator double() const { 
        return col_part + reg_part;
    }
    Energy operator-(const Energy& e) {
        Energy e2 = {.col_part=col_part - e.col_part, .reg_part=reg_part - e.reg_part};
        return e2;
    }
};

struct PotentialMove
{
    Energy delta;
    Block* block;
    Superpixel* new_sp;
    bool operator<(const PotentialMove& other) const {
        return delta < other.delta;
    }
    bool operator>(const PotentialMove& other) const {
        return delta > other.delta;
    }
};

struct Move
{
    Energy energy;
    Point block_index;
    int superpixel_id;
    int num_iterations;
    int level;
};

typedef set<PotentialMove, less<PotentialMove>> MoveSet;

class Engine;

class EnergyEvaluator
{
    unordered_map<int, Energy> sp_cache;
    unordered_map<Block*, Energy> block_cache;
    
public:
    Energy get(Superpixel& superpixel);
    Energy get(Block* block);
    
    virtual Energy calculate(sums& s) = 0;
    
    Energy try_potential_change(Block* block, Superpixel& superpixel);
    void flush(Superpixel& superpixel);
    void flush();
    void flush(Block* block);
};

class CTFEnergyEvaluator : public EnergyEvaluator
{
public:
    virtual Energy calculate(sums& s);
    double e_reg(sums& s);
    double e_col(sums& s);
};


class SegmentationAlgorithm
{
public:
    Engine& engine;
    
    int run_level();
    Energy try_label(Block& block, Superpixel& new_sp);
    
    SegmentationAlgorithm(Engine& engine);
};

class CTFAlgorithm : public SegmentationAlgorithm
{
public:
    int num_iterations;
    CTFEnergyEvaluator energy_evaluator;
    MoveSet get_potential_moves(Block* block);
    optional<PotentialMove> best_move(MoveSet& moves);
    void execute_move(PotentialMove& move);
    bool label_change_violates_minsize(Block* block);
    bool label_change_violates_connectivity(Block* block);
    virtual int run_level();
    Energy try_label(Block& block, Superpixel& new_sp);
    Energy get_initial_energy();
    Energy total_energy;
    vector<Move> moves_log;
    virtual int setup_block_grid(vector<int>& row_dims, vector<int>& col_dims) {
        return 0;
    };
    virtual void assign_initial_superpixels(map<int, vector<Point>>& superpixel_assigments) {};
    CTFAlgorithm(Engine& engine) : SegmentationAlgorithm(engine) {
        total_energy = get_initial_energy();
        num_iterations = 0;
        moves_log = {};
    };
};

class GreedyCTFAlgorithm : public CTFAlgorithm
{
public:
    virtual int run_level();    
};
