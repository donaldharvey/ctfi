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
    
//    if (string(argv[3]) == "orig") {
//        CTFIAlgorithm algo(e);
//        e.setup(&algo);
//    }
//    else {
        CTFIAlgorithm algo2(e);
        e.setup(&algo2);
//    }
    
//    Mat m;
//    Mat m2;
//    e.draw_boundaries(m);
//    e.draw_means(m2);
//    imwrite(string("/me/d/iCtF superpixels/101087 intermediates/iCtF/") + argv[2] + "/out-init.png", m);
//    imwrite(string("/me/d/iCtF superpixels/101087 intermediates/iCtF/") + argv[2] + "/out-init-mean.png", m2);
    ofstream out;
    out.open(e.get_input_basename() + ".log", ios::out);
    out << e.algorithm->total_energy.col_part << "\t" << e.algorithm->total_energy.reg_part << "\t" << 0 << "\t" << e.level << endl;
    
    Mat out_mat;
    e.labels.convertTo(out_mat, CV_16U);
    out_mat += 1;
//    imwrite(e.get_input_basename() + "-init.png", out_mat);
    
    int min_level = 0;
    if (argc > 3) {
        min_level = atoi(argv[3]);
    }
    else {
        min_level = 0;
    }
    
    while (e.level > min_level) {
        e.run_level();
//        e.draw_boundaries(m);
//        e.draw_means(m2);
//        for(auto b: e.blocks) {
//            cout << "(" << b->index << "): " << e.algorithm.label_change_violates_connectivity(b);
//        }
//        imwrite(string("/me/d/iCtF superpixels/101087 intermediates/iCtF/") + argv[2] + "/out-" + to_string(e.level) + ".png", m);
//        imwrite(string("/me/d/iCtF superpixels/101087 intermediates/iCtF/") + argv[2] + "/out-" + to_string(e.level) + "-mean.png", m2);
    }
    
    e.labels.convertTo(out_mat, CV_16U);
    out_mat += 1;
    imwrite(e.get_input_basename() + ".png", out_mat);
    auto &move = e.algorithm->moves_log.back();
    out << move.energy.col_part << "\t" << e.algorithm->moves_log.size() - 1 << "\t" << move.num_iterations << "\t" << move.level << endl;
    out.close();
}


// interesting image in Vitaliy's initial grid algorithm: 101087