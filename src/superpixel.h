#pragma once

#include <opencv2/core/core.hpp>
#include <map>
#include "utils.h"
#include "block.h"

class Engine;

using namespace std;

class Superpixel
{
public:
    map<Point2i, Block*, utils::Point2iCompare> blocks;
    bool operator==(const Superpixel& other) {
        return other.id == id;
    }
    
    void add_block(Block* block);
    void remove_block(Block* block);
    void reset();
    bool contains_block(Block* block);
    
    sums sums;
    
    Engine& engine;
    const int id;
    int initial_size;
    //    vector<Block> blocks;
    Superpixel(Engine& engine, int id, int initial_size);
};