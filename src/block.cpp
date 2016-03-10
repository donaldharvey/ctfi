#include "block.h"
#include "engine.h"

using namespace cv;

Block::Block(Point2i pos, Point2i index, Size size, Engine* engine, int level) : pos(pos), index(index), size(size), engine(engine), level(level) {
    auto r = Rect(pos, size);
    auto M = engine->img(r);
    
    sums = {
        .pos_sum = {0,0},
        .pos2_sum = {0,0},
        .row_col_sum = 0,
        .col_sum = {0,0,0},
        .col2_sum = {0,0,0},
        .area = size.area(),
        .number_blocks = 1,
    };
    
    for(int i = 0; i < M.rows; i++)
    {
        for(int j = 0; j < M.cols; j++) {
            sums.pos_sum += Vec2d(i + pos.y, j + pos.x);
            sums.pos2_sum += Vec2d(pow(i + pos.y, 2), pow(j + pos.x, 2));
            sums.row_col_sum += (i + pos.y) * (j + pos.x);
            auto col = Scalar(M(i,j));
            sums.col_sum = sums.col_sum + col;
            Scalar col2;
            cv::pow(col, 2, col2);
            sums.col2_sum = sums.col2_sum + col2;
        }
    }
};

void Block::split(Mat_<Block*>& m, Point new_idx) {
    superpixel->remove_block(this);
    if (m.rows == 1) {
        if (m.cols == 1) {
            Block* b00 = new Block(*this);
            b00->index = new_idx;
            b00->level -= 1;
            m(0,0) = b00;
            b00->superpixel = superpixel;
            superpixel->add_block(b00);
        }
        else {
            Size new_size_00 = Size(int(size.width/2), size.height);
            Size new_size_01 = Size(size.width - new_size_00.width, size.height);
            auto b00 = new Block(pos, new_idx, new_size_00, engine, level - 1);
            auto b01 = new Block(pos + Point(new_size_00.width, 0), new_idx+Point(1,0), new_size_01, engine, level - 1);
            m(0,0) = b00;
            m(0,1) = b01;
            b00->superpixel = superpixel;
            b01->superpixel = superpixel;
            superpixel->add_block(b00);
            superpixel->add_block(b01);
        }
    }
    else {
        if (m.cols == 1) {
            Size new_size_00 = Size(size.width, int(size.height/2));
            Size new_size_10 = Size(size.width, size.height - new_size_00.height);
            auto b00 = new Block(pos, new_idx, new_size_00, engine, level - 1);
            auto b10 = new Block(pos + Point(0, new_size_00.height), new_idx+Point(0,1), new_size_10, engine, level - 1);
            m(0,0) = b00;
            m(1,0) = b10;
            b00->superpixel = superpixel;
            b10->superpixel = superpixel;
            superpixel->add_block(b00);
            superpixel->add_block(b10);
        }
        else {
            Size new_size_00 = Size(int(size.width/2), int(size.height/2));
            Size new_size_01 = Size(size.width - new_size_00.width, new_size_00.height);
            Size new_size_10 = Size(new_size_00.width, size.height - new_size_00.height);
            Size new_size_11 = Size(size.width - new_size_00.width, size.height - new_size_00.height);
            
            auto b00 = new Block(pos, new_idx, new_size_00, engine, level - 1);
            auto b01 = new Block(pos + Point(new_size_00.width, 0), new_idx+Point(1,0), new_size_01, engine, level - 1);
            auto b10 = new Block(pos + Point(0, new_size_00.height), new_idx+Point(0,1), new_size_10, engine, level - 1);
            auto b11 = new Block(pos + Point(new_size_00.width, new_size_00.height), new_idx+Point(1,1), new_size_11, engine, level - 1);
            
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