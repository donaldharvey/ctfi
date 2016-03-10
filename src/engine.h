#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "superpixel.h"
#include "block.h"
#include "utils.h"
#include "algorithm.h"

#include <boost/optional/optional.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;

template<typename T>
using optional = boost::optional<T>;

const int NUMBER_SUPERPIXELS = 100;
const int MAX_PIXEL_SIZE = 16;

class Engine
{
    void setup_block_grid();
    void update_boundary_blocks();
    void initialize_superpixels();
    void update_boundary_blocks(Block* block);
    void split_blocks();
    
    
    
    vector<int> sp_row_dims;
    vector<int> sp_col_dims;
    int max_block_size;
    int starting_level;

public:
    string input_filename;

    string get_input_basename();
    
    Mat_<bool> is_boundary_block;
    
    int level;
    const Mat orig_img;
    Mat3d img;
    Mat_<Block*> blocks;
    vector<Superpixel> superpixels;
    Mat1i labels;
    CTFAlgorithm* algorithm;
    int number_superpixels; 

    Engine(Mat& orig_img, string& filename, int number_superpixels);

    void setup(CTFAlgorithm* algorithm);
    
    Block* block_at(Point2i idx) {
        if(idx.y < 0 or idx.y >= blocks.rows or idx.x < 0 or idx.x >= blocks.cols) {
            return nullptr;
        }
        return blocks(idx);
    };
    bool is_boundary_block_at(Point2i idx);
    vector<Block*> get_boundary_blocks();
    
    Superpixel& get_superpixel_by_label(int label);
    void relabel(Block* block, Superpixel& superpixel);
    
    Vec3f get_colour(Point2i p);
    
    void draw_boundaries(Mat& output, bool thick_line = false, Vec3b colour = {0, 0, 255});
    void draw_means(Mat& output);
    
    void run();
    void run_level();
    
    Mat_<int32_t> get_segmentation();
};