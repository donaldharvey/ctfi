#include <vector>
#include <iostream>
#include "block.h"
#include "superpixel.h"

using namespace std;
using namespace cv;

namespace utils {
    // From Jian's code
    Mat3d convert_rgb_to_lab(const Mat& img)
    {
        const int RGB2LABCONVERTER_XYZ_TABLE_SIZE = 1024;
        // CIE standard parameters
        const double epsilon = 0.008856;
        const double kappa = 903.3;
        // Reference white
        const double referenceWhite[3] = { 0.950456, 1.0, 1.088754 };
        /// Maximum values
        const double maxXYZValues[3] = { 0.95047, 1.0, 1.08883 };
        
        vector<float> sRGBGammaCorrections(256);
        for (int pixelValue = 0; pixelValue < 256; ++pixelValue) {
            double normalizedValue = pixelValue / 255.0;
            double transformedValue = (normalizedValue <= 0.04045) ? normalizedValue / 12.92 : pow((normalizedValue + 0.055) / 1.055, 2.4);
            
            sRGBGammaCorrections[pixelValue] = transformedValue;
        }
        
        int tableSize = RGB2LABCONVERTER_XYZ_TABLE_SIZE;
        vector<double> xyzTableIndexCoefficients(3);
        xyzTableIndexCoefficients[0] = (tableSize - 1) / maxXYZValues[0];
        xyzTableIndexCoefficients[1] = (tableSize - 1) / maxXYZValues[1];
        xyzTableIndexCoefficients[2] = (tableSize - 1) / maxXYZValues[2];
        
        vector<vector<float> > fXYZConversions(3);
        for (int xyzIndex = 0; xyzIndex < 3; ++xyzIndex) {
            fXYZConversions[xyzIndex].resize(tableSize);
            double stepValue = maxXYZValues[xyzIndex] / tableSize;
            for (int tableIndex = 0; tableIndex < tableSize; ++tableIndex) {
                double originalValue = stepValue*tableIndex;
                double normalizedValue = originalValue / referenceWhite[xyzIndex];
                double transformedValue = (normalizedValue > epsilon) ? pow(normalizedValue, 1.0 / 3.0) : (kappa*normalizedValue + 16.0) / 116.0;
                
                fXYZConversions[xyzIndex][tableIndex] = transformedValue;
            }
        }
        
        Mat3d result = Mat3d(img.rows, img.cols);
        
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                const Vec3b& rgbColor = img.at<Vec3b>(y, x);
                Vec3d& labColor = result(y, x);
                
                float correctedR = sRGBGammaCorrections[rgbColor[2]];
                float correctedG = sRGBGammaCorrections[rgbColor[1]];
                float correctedB = sRGBGammaCorrections[rgbColor[0]];
                float xyzColor[3];
                
                xyzColor[0] = correctedR*0.4124564f + correctedG*0.3575761f + correctedB*0.1804375f;
                xyzColor[1] = correctedR*0.2126729f + correctedG*0.7151522f + correctedB*0.0721750f;
                xyzColor[2] = correctedR*0.0193339f + correctedG*0.1191920f + correctedB*0.9503041f;
                
                int tableIndexX = static_cast<int>(xyzColor[0] * xyzTableIndexCoefficients[0] + 0.5);
                int tableIndexY = static_cast<int>(xyzColor[1] * xyzTableIndexCoefficients[1] + 0.5);
                int tableIndexZ = static_cast<int>(xyzColor[2] * xyzTableIndexCoefficients[2] + 0.5);
                
                float fX = fXYZConversions[0][tableIndexX];
                float fY = fXYZConversions[1][tableIndexY];
                float fZ = fXYZConversions[2][tableIndexZ];
                
                labColor[0] = 116.0*fY - 16.0;
                labColor[1] = 500.0*(fX - fY);
                labColor[2] = 200.0*(fY - fZ);
            }
        }
        return result;
    }
    class UnionFind {
    protected:
        std::vector<int> vector;
    public:
        UnionFind(int size)
        {
            vector.resize(size, -1);    // -1 is size of each
        }
        
        void Union(int e1, int e2)
        {
            e1 = Find(e1);
            e2 = Find(e2);
            
            if (e1 == e2)
                return;
            
            int size1 = -vector[e1];
            int size2 = -vector[e2];
            
            if (size1 < size2) { vector[e1] = e2; vector[e2] = -(size1 + size2); }
            else { vector[e2] = e1; vector[e1] = -(size1 + size2); }
        }
        
        inline int Find(int e)
        {
            int root = e;
            
            while (vector[root] >= 0) root = vector[root];
            if (e != root) {
                while (vector[e] != root) {
                    int tmpe = vector[e];
                    vector[e] = root;
                    e = tmpe;
                }
            }
            return root;
        }
        
        inline int Size(int e) 
        {
            return -vector[Find(e)];
        }
        
    };

    vector<int> grid_partition(int length, int number_items) {
        int n_each = length / number_items;
        int rem = length % number_items;
        n_each += rem / number_items;
        rem = rem % number_items;
        vector<int> v(number_items, n_each);
        if (rem) {
            for(int i=0; i<rem; i++) {
                v[i * rem/number_items] += 1;
            }
        }
        return v;
    }
    
    int patch_3x3_to_bits(const Mat_<Block*>& m, int ulr, int ulc, Block* p)
    {
        int patch = 0;
        int bit = 1;
        const Block* q;
        
        for (int r = ulr; r < ulr + 3; r++) {
            if (r >= m.rows)
                break;
            if (r < 0) {
                bit <<= 3;
            } else {
                for (int c = ulc; c < ulc + 3; c++) {
                    if (c >= 0 && c < m.cols) {
                        q = m(r, c);
                        if (p->index != q->index && q->superpixel->id == p->superpixel->id) patch |= bit;
                    }
                    bit <<= 1;
                }
            }
        }
        return patch;
    }
    
    Mat_<int> bits_to_patch_3x3(int b)
    {
        Mat_<int> patch(3, 3);
        int bit = 1;
        
        for (int i = 0; i < 9; i++) {
            patch(i / 3, i % 3) = ((bit & b) == 0) ? 0 : 1;
            bit <<= 1;
        }
        return patch;
    }
    
    bool is_patch_3x3_connected(int bits)
    {
        Mat_<int> patch = bits_to_patch_3x3(bits);
        int rows = 3;
        int cols = 3;
        UnionFind uf(rows*cols);
        cv::Mat1i comp = cv::Mat1i(rows + 1, cols + 1, -1);
        int cCount = 0;
        
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int q = patch(r, c);
                
                if (q == 1) {
                    int topi = comp(r, c + 1);
                    int lefti = comp(r + 1, c);
                    
                    if (topi < 0) {
                        if (lefti < 0) comp(r + 1, c + 1) = cCount++;
                        else comp(r + 1, c + 1) = lefti;
                    }
                    else { // topi >= 0
                        if (lefti < 0) comp(r + 1, c + 1) = topi;
                        else {
                            comp(r + 1, c + 1) = lefti;
                            uf.Union(lefti, topi);
                        }
                    }
                }
            }
        }
        return uf.Size(0) == cCount;
    }

    // Cached info for connectivity of 3x3 patches
    class ConnectivityCache {
        vector<bool> cache;
    public:
        ConnectivityCache();
        bool IsConnected(int b) const { return cache[b]; }
    private:
        void Initialize();
    };

    
    void ConnectivityCache::Initialize()
    {
        cache.resize(512);
        for (int i = 0; i < 512; i++) {
            cache[i] = is_patch_3x3_connected(i);
        }
    }
    
    ConnectivityCache::ConnectivityCache()
    {
        Initialize();
    }
    
    static ConnectivityCache connectivityCache;
    
    // Return true if superpixel sp is connected in region defined by upper left/lower right corners of pixelsImg
    // Corners are adjusted to be valid for image but we assume that lrr >= 0 and lrc >= 0 and ulr < pixelsImg.rows and ulc < pixelsImg.cols
    bool check_connectivity(const Mat_<Block*>& blocks, Block* p, int ulr, int ulc, int lrr, int lrc)
    {
        if (ulr < 0) ulr = 0;
        if (ulc < 0) ulc = 0;
        if (lrr >= blocks.rows) lrr = blocks.rows - 1;
        if (lrc >= blocks.cols) lrc = blocks.cols - 1;
        
        int rows = lrr - ulr;
        int cols = lrc - ulc;
        UnionFind uf(rows * cols);
        cv::Mat1i comp = cv::Mat1i(rows + 1, cols + 1, -1);
        int cCount = 0;
        
        for (int r = ulr; r < lrr; r++) {
            int ir = r - ulr + 1;
            
            for (int c = ulc; c < lrc; c++) {
                const Block* q = blocks(r, c);
                
                if (p != q && q->superpixel->id == p->superpixel->id) {
                    int ic = c - ulc + 1;
                    int topi = comp(ir - 1, ic);
                    int lefti = comp(ir, ic - 1);
                    
                    if (topi < 0) {
                        if (lefti < 0) comp(ir, ic) = cCount++;
                        else comp(ir, ic) = lefti;
                    } else { // topi >= 0
                        if (lefti < 0) comp(ir, ic) = topi;
                        else {
                            comp(ir, ic) = lefti;
                            uf.Union(lefti, topi);
                        }
                    }
                }
            }
        }
        return uf.Size(0) == cCount;
    }
    
    // See IsSuperpixelRegionConnected; optimized version which uses pre-calculated connectivity data for 3x3 regions
    bool is_connected_optimized(const Mat_<Block*>& blocks, Block* p, int ulr, int ulc, int lrr, int lrc)
    {
        int bits = patch_3x3_to_bits(blocks, ulr, ulc, p);
        return connectivityCache.IsConnected(bits);
    }
    

    
    

}