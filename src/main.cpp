#include "engine.h"
#include "ctfi.h"
#include "origctf.h"
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <string>

using namespace cv;
using namespace std;


int main(int argc, const char * argv[]) {
    Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    string filename = string(argv[1]);
    Engine e = Engine(image, filename, atoi(argv[2]));
    
    if (string(argv[4]) == "orig") {
        CTFIAlgorithm algo(e);
        e.setup(&algo);
    }
    else {
        OrigCTFAlgorithm algo(e);
        e.setup(&algo);
    }
    
//    Mat m;
//    Mat m2;
//    e.draw_boundaries(m);
//    e.draw_means(m2);
//    imwrite("/me/s/desk/out9.png", m);
//    imwrite("/me/s/desk/out9-mean.png", m2);
    ofstream out;
    out.open(e.get_input_basename() + ".log", ios::out);
    out << e.algorithm->total_energy.col_part << "\t" << e.algorithm->total_energy.reg_part << "\t" << 0 << "\t" << e.level << endl;
    while (e.level > 0) {
        e.run_level();
//        e.draw_boundaries(m);
//        e.draw_means(m2);
//        for(auto b: e.blocks) {
//            cout << "(" << b->index << "): " << e.algorithm.label_change_violates_connectivity(b);
//        }
//        imwrite("/me/s/desk/out" + to_string(e.level) + ".png", m);
//        imwrite("/me/s/desk/out" + to_string(e.level) + "-mean.png", m2);
    }
    imwrite(e.get_input_basename() + ".png", e.labels);
    for(auto &move : e.algorithm->moves_log) {
        out << move.energy.col_part << "\t" <<move.energy.reg_part << "\t" << move.num_iterations << "\t" << move.level << endl;
    }
    out.close();
}


// interesting image in Vitaliy's initial grid algorithm: 101087