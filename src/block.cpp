#include "block.h"
#include "engine.h"
#include <iostream>

using namespace cv;
using namespace std;

Block::Block(Point2i pos, Point2i index, Size size, Engine* engine, int level) : pos(pos), index(index), size(size), engine(engine), level(level) {
    sums = {
        .pos_sum = {0,0},
        .pos2_sum = {0,0},
        .row_col_sum = 0,
        .col_sum = {0,0,0},
        .col2_sum = {0,0,0},
        .area = 0,
        .number_blocks = 0,
    };
    
    assert(size.width > 0);
    assert(size.height > 0);
    
    int m_h = size.height == 1 ? 1 : 2;
    int m_w = size.width == 1 ? 1 : 2;
    if (m_h * m_w != 1) {
        subblocks.create(m_h, m_w);
        if (m_w == 2) {
            if (m_h == 1) {
                Size new_size_00 = Size(int(size.width/2), size.height);
                Size new_size_01 = Size(size.width - new_size_00.width, size.height);
                subblocks(0,0) = new Block(pos, Point(0,0), new_size_00, engine, level - 1);
                subblocks(0,1) = new Block(pos + Point(new_size_00.width, 0), Point(0,0), new_size_01, engine, level - 1);
            }
            else {
                Size new_size_00 = Size(int(size.width/2), int(size.height/2));
                Size new_size_01 = Size(size.width - new_size_00.width, new_size_00.height);
                Size new_size_10 = Size(new_size_00.width, size.height - new_size_00.height);
                Size new_size_11 = Size(size.width - new_size_00.width, size.height - new_size_00.height);
                
                subblocks(0,0) = new Block(pos, Point(0,0), new_size_00, engine, level - 1);
                subblocks(0,1) = new Block(pos + Point(new_size_00.width, 0), Point(0,0), new_size_01, engine, level - 1);
                subblocks(1,0) = new Block(pos + Point(0, new_size_00.height), Point(0,0), new_size_10, engine, level - 1);
                subblocks(1,1) = new Block(pos + Point(new_size_00.width, new_size_00.height), Point(0,0), new_size_11, engine, level - 1);
            }
        }
        else {
            Size new_size_00 = Size(size.width, int(size.height/2));
            Size new_size_10 = Size(size.width, size.height - new_size_00.height);
            subblocks(0,0) = new Block(pos, index, new_size_00, engine, level - 1);
            subblocks(1,0) = new Block(pos + Point(0, new_size_00.height), Point(0,0), new_size_10, engine, level - 1);
        }
        
        for(auto block: subblocks) {
            sums += block->sums;
        }
        sums.number_blocks = 1;
    }
    else {
        sums.pos_sum = Vec2d(pos.y, pos.x);
        sums.pos2_sum = Vec2d(pow(pos.y, 2), pow(pos.x, 2));
        sums.row_col_sum = (pos.y) * (pos.x);
        sums.col_sum = Scalar(engine->img(pos));
        cv::pow(sums.col_sum, 2, sums.col2_sum);
        sums.area = 1;
        sums.number_blocks = 1;
    }
};

void Block::split(Mat_<Block*>& m, Point new_idx) {
    if (size.area() == 1) {
        Block* b00 = new Block(*this);
        b00->index = new_idx;
        b00->level -= 1;
        m(0,0) = b00;
        b00->superpixel = superpixel;
        superpixel->add_block(b00);
        return;
    }
    if (size.height == 1) {
        m(0,0) = subblocks(0,0);
        m(0,0)->index = new_idx;
        m(0,1) = subblocks(0,1);
        m(0,1)->index = new_idx + Point(1, 0);
        m(0,0)->superpixel = superpixel;
        m(0,1)->superpixel = superpixel;
        superpixel->add_block(m(0,0));
        superpixel->add_block(m(0,1));
    }
    else if (size.width == 1) {
        m(0,0) = subblocks(0,0);
        m(1,0) = subblocks(1,0);
        m(0,0)->index = new_idx;
        m(1,0)->index = new_idx+Point(0,1);
        m(0,0)->superpixel = superpixel;
        m(1,0)->superpixel = superpixel;
        superpixel->add_block(m(0,0));
        superpixel->add_block(m(1,0));
    }
    else {
        auto b00 = subblocks(0,0);
        auto b01 = subblocks(0,1);
        auto b10 = subblocks(1,0);
        auto b11 = subblocks(1,1);
        
        b00->index = new_idx;
        b01->index = new_idx + Point(1,0);
        b10->index = new_idx + Point(0,1);
        b11->index = new_idx + Point(1,1);
        
        m(0,0) = b00;
        m(0,1) = b01;
        m(1,0) = b10;
        m(1,1) = b11;
        
        b00->superpixel = superpixel;
        b01->superpixel = superpixel;
        b10->superpixel = superpixel;
        b11->superpixel = superpixel;
        
        superpixel->add_block(b00);
        superpixel->add_block(b01);
        superpixel->add_block(b10);
        superpixel->add_block(b11);
    }
}

void Block::get_four_neighbourhood(vector<Block*>& v) {
    for (int i=0; i<4; ++i) {
        auto b = engine->block_at(index+FOUR_DELTAS[i]);
        if(b) {
            v[i] = b;
        }
    }
}

void Block::get_differently_labeled_neighbours(vector<Block*>& v) {
    for (int i=0; i<4; ++i) {
        auto b = engine->block_at(index+FOUR_DELTAS[i]);
        if(b && b->superpixel != superpixel) {
            v[i] = b;
        }
    }
}

Rect Block::get_rect() {
    return Rect(pos, size);
}

bool Block::get_is_boundary() {
    return engine->is_boundary_block_at(index);
}