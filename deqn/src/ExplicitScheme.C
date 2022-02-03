#include "ExplicitScheme.h"
#include "Constants.h"
#include <omp.h>
#include <iostream>
#include <cstring>
#include <math.h>
#define POLY2(i, j, imin, jmin, ni) (((i) - (imin)) + (((j)-(jmin)) * (ni)))

ExplicitScheme::ExplicitScheme(const InputFile* input, Mesh* m) :
    mesh(m)
{
    // int nx = mesh->getNx()[0];
    // int ny = mesh->getNx()[1];
}

void ExplicitScheme::doAdvance(const double dt)
{
    double start = omp_get_wtime();
    diffuse(dt);
    double end = omp_get_wtime();

    start = omp_get_wtime();
    reset();
    end = omp_get_wtime();

    start = omp_get_wtime();
    updateBoundaries();
    end = omp_get_wtime();

    #if DEBUG_TIMINGS
    printf("diffuse took time %f microseconds\n", (end - start) * 1e6);
    printf("reset took time %f microseconds\n", (end - start) * 1e6);
    printf("updateBoundaries took time %f microseconds\n", (end - start) * 1e6);
    #endif
}

void ExplicitScheme::updateBoundaries()
{
    reflectBoundaries();    
}

void ExplicitScheme::init()
{
    updateBoundaries();
}

void ExplicitScheme::reset()
{
    double start = omp_get_wtime();
    double* u0 = mesh->getU0();
    double* u1 = mesh->getU1();
    int x_min = mesh->getMin()[0];
    int x_max = mesh->getMax()[0];
    int y_min = mesh->getMin()[1]; 
    int y_max = mesh->getMax()[1]; 

    int nx = mesh->getNx()[0]+2;

//    #pragma omp parallel for firstprivate(u0, u1, x_min, x_max, y_min, y_max, nx) schedule(static)
//    for(int k = y_min-1; k <= y_max+1; k++) {
//        for(int j = x_min-1; j <=  x_max+1; j++) {
//            int i = POLY2(j,k,x_min-1,y_min-1,nx);
//            u0[i] = u1[i];
//        }
//    }
    #pragma omp parallel for firstprivate(x_min, x_max, y_min, y_max, nx) schedule(static)
    for(int k = y_min-1; k <= y_max+1; k++) {
        int start = POLY2(x_min - 1,k,x_min-1,y_min-1,nx);
        int end = POLY2(x_max + 1,k,x_min-1,y_min-1,nx);
        std::memcpy(&u0[start], &u1[start], sizeof(double) * (end - start));
    }
    // mesh->switchMatricies();
    // int *n = mesh->getNx();
    // #pragma omp parallel for firstprivate(n, x_min, x_max, y_min, y_max, nx) schedule(static)
    // for (int k = 0; k < (n[0] + 2) * (n[1] + 2); k++) {
    //     u0[k] = u1[k];
    // }
    // int *n = mesh->getNx();
    // std::memcpy(u1, u0, sizeof(double) * (n[0] + 2) * (n[1] + 2));
}

void ExplicitScheme::diffuse(double dt)
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

    double rx = dt/(dx*dx);
    double ry = dt/(dy*dy);
//    int k, j;

    #pragma omp parallel for firstprivate(nx, y_max, y_min, x_min, x_max, ry, rx) schedule(static)
    for(int k=y_min; k <= y_max; k++) {
        // #pragma omp simd
        for(int j=x_min; j <= x_max; j++) {

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

void ExplicitScheme::reflectBoundaries()
{
    double* u0 = mesh->getU0();
    int x_min = mesh->getMin()[0];
    int x_max = mesh->getMax()[0];
    int y_min = mesh->getMin()[1]; 
    int y_max = mesh->getMax()[1]; 

    int nx = mesh->getNx()[0]+2;

    #pragma omp parallel
    #pragma omp single nowait
    {
         
        /* top */
        #pragma omp task firstprivate(x_min, x_max, y_min, y_max, nx)
        {
            for(int j = x_min; j <= x_max; j++) {
                int n1 = POLY2(j, y_max, x_min-1, y_min-1, nx);
                int n2 = POLY2(j, y_max+1, x_min-1, y_min-1, nx);

                u0[n2] = u0[n1];
            }
        }
    
        /* right */
        #pragma omp task firstprivate(x_min, x_max, y_min, y_max, nx)
        {
            for(int k = y_min; k <= y_max; k++) {
                int n1 = POLY2(x_max, k, x_min-1, y_min-1, nx);
                int n2 = POLY2(x_max+1, k, x_min-1, y_min-1, nx);

                u0[n2] = u0[n1];
            }
        }
        
        /* bottom */
        #pragma omp task firstprivate(x_min, x_max, y_min, y_max, nx)
        {
            for(int j = x_min; j <= x_max; j++) {
                int n1 = POLY2(j, y_min, x_min-1, y_min-1, nx);
                int n2 = POLY2(j, y_min-1, x_min-1, y_min-1, nx);

                u0[n2] = u0[n1];
            }
        }
        
        /* left */
        #pragma omp task firstprivate(x_min, x_max, y_min, y_max, nx)
        {
            for(int k = y_min; k <= y_max; k++) {
                int n1 = POLY2(x_min, k, x_min-1, y_min-1, nx);
                int n2 = POLY2(x_min-1, k, x_min-1, y_min-1, nx);

                u0[n2] = u0[n1];
            }
        }
        // default: std::cerr << "Error in reflectBoundaries(): unknown boundary id (" << boundary_id << ")" << std::endl;
    }
}
