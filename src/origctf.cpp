#include "origctf.h"
#include "engine.h"
#include "superpixel.h"

// Works for x, y > 0
inline int iCeilDiv(int x, int y)
{
    return (x + y - 1) / y;
}

void calcPixelSizes(int actualGridSize, int maxPixelSize,
                    int& actualMaxPixelSize, int& actualMinPixelSize, int& maxN, int& minN)
{
    int actualDiv = iCeilDiv(actualGridSize, maxPixelSize);
    
    actualMaxPixelSize = iCeilDiv(actualGridSize, actualDiv);
    actualMinPixelSize = actualGridSize / actualDiv;
    maxN = actualGridSize % actualDiv;
    minN = actualDiv - maxN;
}

int OrigCTFAlgorithm::setup_block_grid(vector<int>& block_row_dims, vector<int>& block_col_dims) {
    int imageSize = engine.labels.rows * engine.labels.cols;
    int gridSize = (int)sqrt((double)imageSize / engine.number_superpixels);
    int initDiv = max(2, iCeilDiv(gridSize, MAX_PIXEL_SIZE));
    int maxPixelSize = iCeilDiv(gridSize, initDiv);
    
    int imgSPixelsRows = iCeilDiv(engine.labels.rows, gridSize);
    int imgSPixelsCols = iCeilDiv(engine.labels.cols, gridSize);
    
    int imgPixelsRows = initDiv * (engine.labels.rows / gridSize) + iCeilDiv(engine.labels.rows % gridSize, maxPixelSize);
    int imgPixelsCols = initDiv * (engine.labels.cols / gridSize) + iCeilDiv(engine.labels.cols % gridSize, maxPixelSize);
    
    block_row_dims = vector<int>(imgPixelsRows);
    block_col_dims = vector<int>(imgPixelsCols);
    
    rowSDims = vector<int>(imgSPixelsRows, initDiv);
    colSDims = vector<int>(imgSPixelsCols, initDiv);
    
    int maxPS, minPS, maxN, minN;
    int ri = 0, ci = 0;
    
    calcPixelSizes(gridSize, maxPixelSize, maxPS, minPS, maxN, minN);
    while (ri < initDiv * (engine.labels.rows / gridSize)) {
        for (int i = 0; i < maxN; i++) block_row_dims[ri++] = maxPS;
        for (int i = 0; i < minN; i++) block_row_dims[ri++] = minPS;
    }
    while (ci < initDiv * (engine.labels.cols / gridSize)) {
        for (int i = 0; i < maxN; i++) block_col_dims[ci++] = maxPS;
        for (int i = 0; i < minN; i++) block_col_dims[ci++] = minPS;
    }
    if (engine.labels.rows % gridSize > 0) {
        calcPixelSizes(engine.labels.rows % gridSize, maxPixelSize, maxPS, minPS, maxN, minN);
        for (int i = 0; i < maxN; i++) block_row_dims[ri++] = maxPS;
        for (int i = 0; i < minN; i++) block_row_dims[ri++] = minPS;
        rowSDims.back() = maxN + minN;
    }
    if (engine.labels.cols % gridSize > 0) {
        calcPixelSizes(engine.labels.cols % gridSize, maxPixelSize, maxPS, minPS, maxN, minN);
        for (int i = 0; i < maxN; i++) block_col_dims[ci++] = maxPS;
        for (int i = 0; i < minN; i++) block_col_dims[ci++] = minPS;
        colSDims.back() = maxN + minN;
    }
    return maxPixelSize;
}

void OrigCTFAlgorithm::assign_initial_superpixels(map<int, vector<Point>>& superpixel_assignments) {
    int i0, j0;
    
    int px_id = 0;
    i0 = 0;
    for (int pi = 0; pi < rowSDims.size(); pi++) {
        int i1 = i0 + rowSDims[pi];
        
        j0 = 0;
        for (int pj = 0; pj < colSDims.size(); pj++) {
            int j1 = j0 + colSDims[pj];
            vector<Point> v = {};
            
            for (int i = i0; i < i1; i++) {
                for (int j = j0; j < j1; j++) {
                    v.push_back(Point(j,i));
                }
            }
            superpixel_assignments[px_id++] = v;
            j0 = j1;
        }
        i0 = i1;
    }
}