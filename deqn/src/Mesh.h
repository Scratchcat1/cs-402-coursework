#ifndef DIFFUSION_MESH_H_
#define DIFFUSION_MESH_H_

#include "InputFile.h"

class Mesh {
    private:
        const InputFile* input;

        float* u1;
        float* u0;
        float* cellx;
        float* celly;

        float* min_coords;
        float* max_coords;

        int NDIM;

        int* n; 
        int* min;
        int* max;

        float* dx;

        /*
         * A mesh has four neighbours, and they are 
         * accessed in the following order:
         * - top
         * - right
         * - bottom
         * - left
         */
        int* neighbours;

        void allocate();
        bool allocated;
    public:
        Mesh(const InputFile* input);
                
        float* getU0();
        float* getU1();

        float* getDx();
        int* getNx();
        int* getMin();
        int* getMax();
        int getDim();

        float* getCellX();
        float* getCellY();

        int* getNeighbours();

        float getTotalTemperature();
};
#endif
