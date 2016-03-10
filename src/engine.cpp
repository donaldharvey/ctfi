#include "engine.h"
#include "algorithm.h"
#include "origctf.h"
#include "block.h"

#include <cmath>
#include <iostream>
using namespace cv;
using namespace std;

void Engine::setup_block_grid() {
    vector<int> row_dims, col_dims;
    max_block_size = algorithm->setup_block_grid(row_dims, col_dims);
    
    blocks = Mat_<Block*>(int(row_dims.size()), int(col_dims.size()));

    starting_level = int(ceil(log2(max_block_size))) + 1;
    level = starting_level;
    
    int y = 0;
    int i = 0;
    for(int h: row_dims) {
        int x = 0;
        int j = 0;
        for(int w : col_dims) {
            Block* b = new Block(Point2i(x,y), Point2i(j,i), Size(w,h), this, starting_level);
            blocks(Point2i(j,i)) = b;
            ++j;
            x += w;
        }
        ++i;
        y += h;
    }
    return;
}

void Engine::initialize_superpixels() {
    map<int, vector<Point>> assignments = {};
    algorithm->assign_initial_superpixels(assignments);
    for (auto &it: assignments) {
        superpixels.push_back(Superpixel(*this, it.first, 0));
    }
    for (auto &it: assignments) {
        for (auto idx: it.second) {
            auto b = blocks(idx);
            auto ptr = &superpixels[it.first];
            cout << ptr << endl;
            b->superpixel = ptr;
            b->superpixel->add_block(b);
            b->superpixel->initial_size += b->size.area();
            labels(b->get_rect()) = it.first;
        }
    }
}
//     int division = ceil(sqrt(NUMBER_SUPERPIXELS));
//     sp_row_dims = utils::grid_partition(labels.rows, division);
//     sp_col_dims = utils::grid_partition(labels.cols, division);
    
//     int px_id = 0;
//     int row_idx = 0;
//     for(auto it = sp_row_dims.begin(); it != sp_row_dims.end(); ++it) {
//         int h = *it;
//         int col_idx = 0;
//         for(auto it2 = sp_col_dims.begin(); it2 != sp_col_dims.end(); ++it2) {
//             int w = *it2;
//             labels(Rect(col_idx, row_idx, w, h)) = px_id;
            
//             Superpixel sp = Superpixel(*this, px_id, w*h);
//             superpixels.push_back(sp);
//             ++px_id;
//             col_idx += w;
//         }
//         row_idx += h;
//     }
    
// }
// void Engine::populate_blocks() {
//     int blocks_initial_size = ceil(sqrt(NUMBER_SUPERPIXELS))*2;
//     vector<Block> blocks_test = {};
//     blocks = Mat_<Block>(blocks_initial_size, blocks_initial_size);
    
//     vector<int> row_dims, col_dims = {};
//     max_block_size = 0;
    
//     for(auto it = sp_row_dims.begin(); it != sp_row_dims.end(); ++it) {
//         int l = *it / 2;
//         row_dims.push_back(l);
//         row_dims.push_back(l + *it % 2);
//         if (l > max_block_size) {
//             max_block_size = l;
//         }
//     }
    
//     for(auto it = sp_col_dims.begin(); it != sp_col_dims.end(); ++it) {
//         int l = *it / 2;
//         col_dims.push_back(l);
//         col_dims.push_back(l + *it % 2);
//         if (l > max_block_size) {
//             max_block_size = l;
//         }
//     }
    
//     
// }

//void Engine::setup_block_grid() {
//    ifstream input;
//    input.open("/me/s/down/init100/130066mesh100.txt");
//    string item;
//    getline(input, item); // ignore first line
//    
//    
//    getline(input, item, " "); // block dims...
//    int width = stoi(item);
//    // scan to end
//    getline(input, item);
//    vector<int> block_widths = vector<int>(width);
//    int i = 0;
//    while (getline(input, item, ' ')) {
//        if (i == width) {
//            break;
//        }
//        block_widths.push_back(stoi(item));
//        ++i;
//    }
//    getline(input, item);
//    
//    
//    getline(input, item, ' ');
//    
//    int height = stoi(item);
//    
//    getline(input, item);
//    vector<int> block_heights = vector<int>(height);
//
//    while (getline(input, item, ' ')) {
//        if (i == height) {
//            break;
//        }
//        block_heights.push_back(stoi(item));
//        ++i;
//    }
//    vector<Point> superpixel_blocks = {};
//    getline(input, item);
//    
//    getline(input, item);
//    
//    string line = item;
//    getline(input, item, " ");
//    
//    while(getline(input, item)) {
//        istringstream ss(item);
//        string token;
//        string token2;
//        while(std::getline(ss, token, ' ')) {
//            std::getline(ss, token2, ' ');
//            auto p = Point(stoi(token), stoi(token2));
//            superpixel_blocks.push_back(p);
//        }
//    }
//    
//    
//
//}

void Engine::split_blocks() {
    int row_sum = 0, col_sum = 0;
    for (Block* block : blocks(Rect(0,0,1,blocks.rows))) {
        row_sum += (block->size.height == 1 ? 1 : 2);
    }
    for (Block* block : blocks(Rect(0,0,blocks.cols, 1))) {
        col_sum += (block->size.width == 1 ? 1 : 2);
    }
    Mat_<Block*> new_blocks(row_sum, col_sum);
    
    
    int new_row = 0;
    for (int i = 0; i < blocks.rows; ++i) {
        int new_col = 0;
        int new_h = (blocks(i,0)->size.height == 1 ? 1 : 2);
        for (int j = 0; j < blocks.cols; ++j) {
            Block* block = blocks(i,j);
            int new_w = (block->size.width == 1 ? 1 : 2);
            auto m = new_blocks(Rect(new_col, new_row, new_w, new_h));
            block->split(m, Point(new_col, new_row));
            delete block;
            blocks(i,j) = block = nullptr;
            new_col += new_w;
        }
        new_row += new_h;
    }
    blocks = new_blocks;
}

void Engine::update_boundary_blocks() {
    is_boundary_block = Mat_<bool>(blocks.rows, blocks.cols, false);
    for (int i = 0; i < blocks.rows; ++i)
    {
        // const Block* ptr = blocks.ptr<Block>(i);  // FIXME work out why not working.
        Block** row_start = blocks[i];
        for (int j = 0; j < blocks.cols; ++j)
        {
            Block* b = *(row_start + j);
//            Block b = blocks(i, j);
//            Block b2 = blocks.
            assert(b->level == this->level);
            update_boundary_blocks(b);
            
        }
    }
}

inline void Engine::update_boundary_blocks(Block* block) {
    auto four_nh = vector<Block*>(4, nullptr);
    block->get_four_neighbourhood(four_nh);
    for (auto &other : four_nh) {
        if (!other) {
            continue;
        }
        if (other->superpixel->id != block->superpixel->id) {
            is_boundary_block(block->index) = true;
            is_boundary_block(other->index) = true;
        }
    }
}

void Engine::run() {
    while (level) {
        run_level();
    }
}

void Engine::run_level() {
    algorithm->run_level();
    if (level > 0) {
        split_blocks();
        level--;
        update_boundary_blocks();
    }
}

bool Engine::is_boundary_block_at(Point2i idx) {
    return is_boundary_block(idx);
}

void Engine::relabel(Block* block, Superpixel &superpixel) {
    assert(block->level == this->level);
    block->superpixel->remove_block(block);
    superpixel.add_block(block);
    block->superpixel = &superpixel;
    labels(block->get_rect()) = superpixel.id;
    update_boundary_blocks(block);
}

Superpixel& Engine::get_superpixel_by_label(int label) {
    return superpixels[label];
}

vector<Block*> Engine::get_boundary_blocks() {
    vector<Block*> v = {};
    for (int i = 0; i < is_boundary_block.rows; ++i) {
        for (int j = 0; j < is_boundary_block.cols; ++j) {
            if (is_boundary_block(i,j)) {
                v.push_back(blocks(i,j));
            }
        }
    }
    return v;
}

void Engine::draw_means(Mat& output) {
    orig_img.copyTo(output);
    vector<cv::Vec3d> means(this->superpixels.size(), 0.0);
    vector<int> counts(this->superpixels.size(), 0);
    for(int p_y = 0; p_y < labels.rows; ++p_y) {
        for(int p_x = 0; p_x < labels.cols; ++p_x) {
            means[labels(p_y, p_x)] += orig_img.at<Vec3b>(p_y, p_x);
            counts[labels(p_y, p_x)]++;
        }
    }
    for(int i=0; i<means.size(); ++i) {
        means[i] /= counts[i];
    }
    for(int p_y = 0; p_y < labels.rows; ++p_y) {
        for(int p_x = 0; p_x < labels.cols; ++p_x) {
            output.at<cv::Vec3b>(p_y, p_x) = means[labels(p_y, p_x)];
        }
    }
}

void Engine::draw_boundaries(Mat& output, bool thick_line, Vec3b colour) {
    orig_img.copyTo(output);
//    Mat1b isBoundary = Mat1b(output.rows, output.cols, 0.0);
//    for (int j = 0; j < blocks.rows; j++)
//    {
//        for (int k = 0; k < blocks.cols; k++)
//        {
//            if (is_boundary_block_at(Point(k, j))) {
////                cout << blocks(j,k)->get_rect() << endl;
//                output(blocks(j,k)->get_rect()) = Vec3b(colour);
//            }
//        }
//    }
    
    const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    
    auto mask = Mat_<uchar>(output.rows, output.cols, 0.0);
    
    for (int j = 0; j < output.rows; j++)
    {
        for (int k = 0; k < output.cols; k++)
        {
            int neighbors = 0;
            for (int i = 0; i < 8; i++)
            {
                int x = k + dx8[i];
                int y = j + dy8[i];
                
                if( (x >= 0 && x < mask.cols) && (y >= 0 && y < mask.rows) )
                {
                    if( labels[y][x] != labels[j][k] )
                    {
                        if( thick_line || !*mask.ptr(y, x) )
                            neighbors++;
                    }
                }
            }
            if( neighbors > 1 )
                *mask.ptr(j, k) = (uchar)255;
        }
    }
    for (int j = 0; j < output.rows; j++)
    {
        for (int k = 0; k < output.cols; k++) {
            if (mask(j,k))
                output.at<Vec3b>(j,k) = colour;
        }
    }
}

string Engine::get_input_basename() {
    boost::filesystem::path p(input_filename);
    return p.stem().string();
}

Engine::Engine(Mat& orig_img, string& filename, int number_superpixels) : orig_img(orig_img), input_filename(filename), number_superpixels(number_superpixels) {
    img = utils::convert_rgb_to_lab(orig_img);  
    labels = Mat1i(orig_img.rows, orig_img.cols, 0);
}

void Engine::setup(CTFAlgorithm* algo) {
    algorithm = algo;
    setup_block_grid();
    initialize_superpixels();
    update_boundary_blocks();
}