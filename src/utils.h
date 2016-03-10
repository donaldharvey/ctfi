#include <vector>
#include <opencv2/core/core.hpp>

#pragma once

using namespace std;
using namespace cv;

class Block;
class Superpixel;

namespace utils {

    vector<int> grid_partition(int length, int number_items);

    struct Point2iCompare {
        inline bool operator()(Point2i a, Point2i b) const {
            if (a.y == b.y) {
                return a.x < b.x;
            }
            return a.y < b.y;
        }
    };
    
    bool is_connected_optimized(const Mat_<Block*>& blocks, Block* p, int ulr, int ulc, int lrr, int lrc);
    
    Mat3d convert_rgb_to_lab(const Mat& img);
}

struct sums {
    Vec2d pos_sum;
    Vec2d pos2_sum;
    double row_col_sum;
    Scalar col_sum;
    Scalar col2_sum;
    int number_blocks;
    int area;
    
    sums operator+(sums &s) {
        sums s2 = {
            .pos_sum = pos_sum + s.pos_sum,
            .pos2_sum = pos2_sum + s.pos2_sum,
            .row_col_sum = row_col_sum + s.row_col_sum,
            .col_sum = col_sum + s.col_sum,
            .col2_sum = col2_sum + s.col2_sum,
            .number_blocks = number_blocks + s.number_blocks,
            .area = area + s.area,
        };
        return s2;
    }
    
    void operator+=(sums &s) {
        pos_sum += s.pos_sum;
        pos2_sum += s.pos2_sum;
        row_col_sum += s.row_col_sum;
        col_sum += s.col_sum;
        col2_sum += s.col2_sum;
        number_blocks += s.number_blocks;
        area += s.area;
    }
    
    void operator-=(sums &s) {
        pos_sum -= s.pos_sum;
        pos2_sum -= s.pos2_sum;
        row_col_sum -= s.row_col_sum;
        col_sum -= s.col_sum;
        col2_sum -= s.col2_sum;
        number_blocks -= s.number_blocks;
        area -= s.area;
    }
    
    sums operator-(sums &s) {
        sums s2 = {
            .pos_sum = pos_sum - s.pos_sum,
            .pos2_sum = pos2_sum - s.pos2_sum,
            .row_col_sum = row_col_sum - s.row_col_sum,
            .col_sum = col_sum - s.col_sum,
            .col2_sum = col2_sum - s.col2_sum,
            .number_blocks = number_blocks - s.number_blocks,
            .area = area - s.area,
        };
        return s2;
    }
};
