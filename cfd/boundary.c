#include <stdio.h>
#include <string.h>
#include "datadef.h"
#include "tiles.h"

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

/* Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
void apply_boundary_conditions(float **u, float **v, char **flag,
    int imax, int jmax, float ui, float vi)
{
    int i, j;

    for (j=0; j<=jmax+1; j++) {
        /* Fluid freely flows in from the west */
        u[0][j] = u[1][j];
        v[0][j] = v[1][j];

        /* Fluid freely flows out to the east */
        u[imax][j] = u[imax-1][j];
        v[imax+1][j] = v[imax][j];
    }

    for (i=0; i<=imax+1; i++) {
        /* The vertical velocity approaches 0 at the north and south
         * boundaries, but fluid flows freely in the horizontal direction */
        v[i][jmax] = 0.0;
        u[i][jmax+1] = u[i][jmax];

        v[i][0] = 0.0;
        u[i][0] = u[i][1];
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the u and v velocity to
     * tend towards zero in these cells.
     */
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax; j++) {
            if (flag[i][j] & B_NSEW) {
                switch (flag[i][j]) {
                    case B_N: 
                        v[i][j]   = 0.0;
                        u[i][j]   = -u[i][j+1];
                        u[i-1][j] = -u[i-1][j+1];
                        break;
                    case B_E: 
                        u[i][j]   = 0.0;
                        v[i][j]   = -v[i+1][j];
                        v[i][j-1] = -v[i+1][j-1];
                        break;
                    case B_S:
                        v[i][j-1] = 0.0;
                        u[i][j]   = -u[i][j-1];
                        u[i-1][j] = -u[i-1][j-1];
                        break;
                    case B_W: 
                        u[i-1][j] = 0.0;
                        v[i][j]   = -v[i-1][j];
                        v[i][j-1] = -v[i-1][j-1];
                        break;
                    case B_NE:
                        v[i][j]   = 0.0;
                        u[i][j]   = 0.0;
                        v[i][j-1] = -v[i+1][j-1];
                        u[i-1][j] = -u[i-1][j+1];
                        break;
                    case B_SE:
                        v[i][j-1] = 0.0;
                        u[i][j]   = 0.0;
                        v[i][j]   = -v[i+1][j];
                        u[i-1][j] = -u[i-1][j-1];
                        break;
                    case B_SW:
                        v[i][j-1] = 0.0;
                        u[i-1][j] = 0.0;
                        v[i][j]   = -v[i-1][j];
                        u[i][j]   = -u[i][j-1];
                        break;
                    case B_NW:
                        v[i][j]   = 0.0;
                        u[i-1][j] = 0.0;
                        v[i][j-1] = -v[i-1][j-1];
                        u[i][j]   = -u[i][j+1];
                        break;
                }
            }
        }
    }

    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    v[0][0] = 2*vi-v[1][0];
    for (j=1;j<=jmax;j++) {
        u[0][j] = ui;
        v[0][j] = 2*vi-v[1][j];
    }
}

/* Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.

 * Note: this is the same function as above but all operations are performed within the tile
 */
void apply_tile_boundary_conditions(float **u, float **v, char **flag,
    int imax, int jmax, float ui, float vi, struct TileData* tile_data)
{
    #pragma omp parallel firstprivate(u, v, flag, imax, jmax, ui, vi, tile_data) default(none)
    {
        int i, j;
        #pragma omp for
        for (j=max(0, tile_data->start_y); j<=min(jmax+1, tile_data->end_y - 1); j++) {
            /* Fluid freely flows in from the west */
            u[0][j] = u[1][j];
            v[0][j] = v[1][j];

            /* Fluid freely flows out to the east */
            u[imax][j] = u[imax-1][j];
            v[imax+1][j] = v[imax][j];
        }

        #pragma omp for
        for (i=max(0, tile_data->start_x); i<=min(imax+1, tile_data->end_x - 1); i++) {
            /* The vertical velocity approaches 0 at the north and south
            * boundaries, but fluid flows freely in the horizontal direction */
            v[i][jmax] = 0.0;
            u[i][jmax+1] = u[i][jmax];

            v[i][0] = 0.0;
            u[i][0] = u[i][1];
        }

        /* Apply no-slip boundary conditions to cells that are adjacent to
        * internal obstacle cells. This forces the u and v velocity to
        * tend towards zero in these cells.
        */
        #pragma omp for collapse(2)
        for (i=max(1, tile_data->start_x); i<=min(imax+1, tile_data->end_x-1); i++) {
            for (j=max(1, tile_data->start_y); j<=min(jmax, tile_data->end_y-1); j++) {
                if (flag[i][j] & B_NSEW) {
                    switch (flag[i][j]) {
                        case B_N: 
                            v[i][j]   = 0.0;
                            u[i][j]   = -u[i][j+1];
                            u[i-1][j] = -u[i-1][j+1];
                            break;
                        case B_E: 
                            u[i][j]   = 0.0;
                            v[i][j]   = -v[i+1][j];
                            v[i][j-1] = -v[i+1][j-1];
                            break;
                        case B_S:
                            v[i][j-1] = 0.0;
                            u[i][j]   = -u[i][j-1];
                            u[i-1][j] = -u[i-1][j-1];
                            break;
                        case B_W: 
                            u[i-1][j] = 0.0;
                            v[i][j]   = -v[i-1][j];
                            v[i][j-1] = -v[i-1][j-1];
                            break;
                        case B_NE:
                            v[i][j]   = 0.0;
                            u[i][j]   = 0.0;
                            v[i][j-1] = -v[i+1][j-1];
                            u[i-1][j] = -u[i-1][j+1];
                            break;
                        case B_SE:
                            v[i][j-1] = 0.0;
                            u[i][j]   = 0.0;
                            v[i][j]   = -v[i+1][j];
                            u[i-1][j] = -u[i-1][j-1];
                            break;
                        case B_SW:
                            v[i][j-1] = 0.0;
                            u[i-1][j] = 0.0;
                            v[i][j]   = -v[i-1][j];
                            u[i][j]   = -u[i][j-1];
                            break;
                        case B_NW:
                            v[i][j]   = 0.0;
                            u[i-1][j] = 0.0;
                            v[i][j-1] = -v[i-1][j-1];
                            u[i][j]   = -u[i][j+1];
                            break;
                    }
                }
            }
        }

        /* Finally, fix the horizontal velocity at the  western edge to have
        * a continual flow of fluid into the simulation.
        */
        v[0][0] = 2*vi-v[1][0];
        #pragma omp for
        for (j=max(1, tile_data->start_y);j<=min(jmax, tile_data->end_y - 1);j++) {
            u[0][j] = ui;
            v[0][j] = 2*vi-v[1][j];
        }
    }
}
