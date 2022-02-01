#include "ExplicitSchemeTiles.h"
#include <omp.h>
#include <iostream>
#include <cstring>
#include <math.h>
#define POLY2(i, j, imin, jmin, ni) (((i) - (imin)) + (((j)-(jmin)) * (ni)))

ExplicitSchemeTiles::ExplicitSchemeTiles(const InputFile* input, Mesh* m, int t_size) : ExplicitScheme(input, m), tile_size(t_size)
{
    // mesh = m;
    // int nx = mesh->getNx()[0];
    // int ny = mesh->getNx()[1];
}

void ExplicitSchemeTiles::diffuse(double dt)
{
    double* u0 = mesh->getU0();
    double* u1 = mesh->getU1();
    int x_min = mesh->getMin()[0];
    int x_max = mesh->getMax()[0];
    int y_min = mesh->getMin()[1]; 
    int y_max = mesh->getMax()[1]; 
    double dx = mesh->getDx()[0];
    double dy = mesh->getDx()[1];

    int nx = mesh->getNx()[0]+2;
    int ny = mesh->getNx()[1]+2;

    double rx = dt/(dx*dx);
    double ry = dt/(dy*dy);

    std::cout << "Tile size of" << tile_size << std::endl;
    int tiles_x = (int) std::ceil((double) (nx - 2) / ((double) tile_size));
    int tiles_y = (int) std::ceil((double) (ny - 2) / ((double) tile_size));
    std::cout << "Tiles in shape " << tiles_x << "x" << tiles_y << std::endl;
//    int k, j;

    #pragma omp parallel for firstprivate(tile_size, tiles_x, tiles_y, nx, y_max, y_min, x_min, x_max, ry, rx) schedule(static)
    for(int tile_num=0; tile_num < (tiles_x * tiles_y); tile_num++) {
        int tile_y = tile_num / tiles_x;
        int tile_x = tile_num % tiles_x;

        int x_range = x_max - x_min;
        int y_range = y_max - y_min;
        int tile_x_min = x_min + tile_size * tile_x;
        int tile_x_max = std::min(tile_x_min + tile_size, x_max);
        int tile_y_min = y_min + tile_size * tile_y; 
        int tile_y_max = std::min(tile_y_min + tile_size, y_max);
        // std::cout << tile_x_min << "," << tile_x_max << "|" << tile_y_min << "," << tile_y_max << std::endl;
        for (int k=tile_y_min; k <= tile_y_max; k++){
            for(int j=tile_x_min; j <= tile_x_max; j++) {
                int n1 = POLY2(j,k,x_min-1,y_min-1,nx);
                int n2 = POLY2(j-1,k,x_min-1,y_min-1,nx);
                int n3 = POLY2(j+1,k,x_min-1,y_min-1,nx);
                int n4 = POLY2(j,k-1,x_min-1,y_min-1,nx);
                int n5 = POLY2(j,k+1,x_min-1,y_min-1,nx);

                u1[n1] = (1.0-2.0*rx-2.0*ry)*u0[n1] + rx*u0[n2] + rx*u0[n3]
                    + ry*u0[n4] + ry*u0[n5];
            }
        }
    }
}
