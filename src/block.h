#pragma once

#include <opencv2/core/core.hpp>
#include "utils.h"
#include <memory>

using namespace cv;
using namespace std;

class Engine;
class Superpixel;

vector<Point2i> const FOUR_DELTAS = {Point2i(-1,0), Point2i(1,0), Point2i(0,1), Point2i(0,-1)};

class Block
{
public:
    Point pos;
    Point index;
    Size size;
    int level;
    
    sums sums;
    Engine* engine;
    Superpixel* superpixel;
    
    void get_four_neighbourhood(vector<Block*>& v);
    void get_differently_labeled_neighbours(vector<Block*>& v);
    Rect get_rect();
    void split(Mat_<Block*>& m, Point new_idx);
    bool get_is_boundary();
    void relabel(Superpixel& new_sp);
    
    Block(Point2i pos, Point2i index, Size size, Engine* engine, int level);
};

