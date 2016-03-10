#include "ctfi.h"
#include <fstream>
#include <iostream>
#include "engine.h"

using namespace std;

void Read_Block_Sizes(String line, vector<int>& sizes)
{
    istringstream iss( line );
    int current, previous;
    iss >> previous;
    for ( int i = 0; i < sizes.size(); i++ )
    {
        iss >> current;
        sizes[i] = current - previous;
        previous = current;
    }
}

void Read_Mesh (String name, vector<int>& widths, vector<int>&
                heights, vector< vector<Point> >& superpixels)
{
    ifstream file;
    file.open( name );
    widths.clear();
    heights.clear();
    superpixels.clear();
    if ( ! file.is_open() ) { cout << "\nError opening " << name; return; }
    string line;
    getline( file, line ); // read line 1 with width height
    getline( file, line ); // read line 2 with the number of blocks in the horizontal direction
    istringstream stream_w( line );  // access line as a stream
    int entry;
    stream_w >> entry;
    widths.resize( entry );
    getline( file, line ); // read line 3 with integer x-coordinates of bounds between blocks
    istringstream stream_x( line );
    Read_Block_Sizes( line, widths );
    getline( file, line ); // read line 4 with the number of blocks in the horizontal direction
    istringstream stream_h( line );
    stream_h >> entry;
    heights.resize( entry );
    getline( file, line ); // read line 5 with integer x-coordinates of bounds between blocks
    istringstream stream_y( line );
    Read_Block_Sizes( line, heights );
    getline( file, line ); // read line 6 with the number of superpixels
    istringstream stream_s( line );
    stream_s >> entry;
    superpixels.resize( entry );
    Point p;
    bool first = true;
    for ( int i = 0; i < superpixels.size(); i++ )
    {
        getline( file, line ); // read line i+7 with the integer x y indices of blocks in the i-th superpixel
        istringstream stream( line );
        while( stream >> entry )
            if ( first ) { p.y = entry; first = false; }
            else { p.x = entry; first = true; superpixels[i].push_back( p ); }
    }
}

int CTFIAlgorithm::setup_block_grid(vector<int>& block_row_dims, vector<int>& block_col_dims) {
    Read_Mesh("/me/s/down/init" + to_string(engine.number_superpixels) + "/" + engine.get_input_basename() + "mesh" + to_string(engine.number_superpixels) + ".txt", block_col_dims, block_row_dims, assignments);
    int max = 0;
    for(auto i: block_row_dims) {
        if (i > max) {
            max = i;
        }
    }
    for(auto i: block_col_dims) {
        if (i > max) {
            max = i;
        }
    }
    return max;
}

void CTFIAlgorithm::assign_initial_superpixels(map<int, vector<Point>>& superpixel_assignments) {
    for(int i=0; i<assignments.size(); ++i) {
        superpixel_assignments[i] = assignments[i];
    }
}