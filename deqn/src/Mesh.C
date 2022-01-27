#include "Mesh.h"

#include <cstdlib>
#include <iostream>

#define POLY2(i, j, imin, jmin, ni) (((i) - (imin)) + ((j)-(jmin)) * (ni))

Mesh::Mesh(const InputFile* input):
    input(input)
{
    allocated = false;

    NDIM = 2;

    n = new int[NDIM];
    min = new int[NDIM];
    max = new int[NDIM];
    dx = new float[NDIM];

    int nx = input->getInt("nx", 0);
    int ny = input->getInt("ny", 0);

    min_coords = new float[NDIM];
    max_coords = new float[NDIM];

    min_coords[0] = input->getfloat("xmin", 0.0);
    max_coords[0] = input->getfloat("xmax", 1.0);
    min_coords[1] = input->getfloat("ymin", 0.0);
    max_coords[1] = input->getfloat("ymax", 1.0);

    // setup first dimension.
    n[0] = nx;
    min[0] = 1;
    max[0] = nx;

    dx[0] = ((float) max_coords[0]-min_coords[0])/nx;

    // setup second dimension.
    n[1] = ny;
    min[1] = 1;
    max[1] = ny;

    dx[1] = ((float) max_coords[1]-min_coords[1])/ny;
    
    allocate();
}

void Mesh::allocate()
{
    allocated = true;

    int nx = n[0];
    int ny = n[1];

    /* Allocate arrays */
    u1 = new float[(nx+2) * (ny+2)];
    u0 = new float[(nx+2) * (ny+2)];
    
    /* Allocate and initialise coordinate arrays */
    cellx = new float[nx+2];
    celly = new float[ny+2];

    float xmin = min_coords[0];
    float ymin = min_coords[1];

    for (int i=0; i < nx+2; i++) {
        cellx[i]=xmin+dx[0]*(i-1);
    }

    for (int i = 0; i < ny+2; i++) {
        celly[i]=ymin+dx[1]*(i-1);
    }
}

float* Mesh::getU0()
{
    return u0;
}

float* Mesh::getU1()
{
    return u1;
}

float* Mesh::getDx()
{
    return dx;
}

int* Mesh::getMin()
{
    return min;
}

int* Mesh::getMax()
{
    return max;
}

int Mesh::getDim()
{
    return NDIM;
}

int* Mesh::getNx()
{
    return n;
}

int* Mesh::getNeighbours()
{
    return neighbours;
}

float* Mesh::getCellX()
{
    return cellx;
}

float* Mesh::getCellY()
{
    return celly;
}

float Mesh::getTotalTemperature()
{
    if(allocated) {
        float temperature = 0.0;
        int x_min = min[0];
        int x_max = max[0];
        int y_min = min[1]; 
        int y_max = max[1]; 

        int nx = n[0]+2;

        for(int k=y_min; k <= y_max; k++) {
            for(int j=x_min; j <= x_max; j++) {

                int n1 = POLY2(j,k,x_min-1,y_min-1,nx);

                temperature += u0[n1];
            }
        }

        return temperature;
    } else {
        return 0.0;
    }
}
