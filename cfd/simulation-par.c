#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "datadef.h"
#include "init.h"
#include "tiles.h"
#include <mpi.h>
#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

extern int *ileft, *iright;
extern int nprocs, proc;

/////////////////////////////////////////////
// Note: all the loops here iterate over the tile
// rather than the full array
// Conversion is simple 
// start -> max(start, tile->start_?)
// end -> min(end, tile->end_?)
// As this appears all through the program
// I will only comment about it here to avoid
// repeating everything a bunch of times
/////////////////////////////////////////////


/* Computation of tentative velocity field (f, g) */
void compute_tentative_velocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re, struct TileData* tile_data, double * sync_time_taken)
{
    // Create the threads outside the two big loops to avoid the overhead of creating and joining the threads twice
    // Use firstprivate to provide a private copy of the parameters
    #pragma omp parallel firstprivate(imax, jmax, del_t, delx, dely, gamma, Re, tile_data, u, v, f, g, flag) default(none)
    {
    int  i, j;
    float du2dx, duvdy, duvdx, dv2dy, laplu, laplv;
    // Start a parallel for loop
    #pragma omp for collapse(2)
    for (i=max(1, tile_data->start_x); i<=min(tile_data->end_x-1,imax-1); i++) { // i=1 i <=imax -1
        for (j=max(1, tile_data->start_y); j<=min(tile_data->end_y-1, jmax); j++) { // j=1 j <=jmax
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                du2dx = ((u[i][j]+u[i+1][j])*(u[i][j]+u[i+1][j])+
                    gamma*fabs(u[i][j]+u[i+1][j])*(u[i][j]-u[i+1][j])-
                    (u[i-1][j]+u[i][j])*(u[i-1][j]+u[i][j])-
                    gamma*fabs(u[i-1][j]+u[i][j])*(u[i-1][j]-u[i][j]))
                    /(4.0*delx);
                duvdy = ((v[i][j]+v[i+1][j])*(u[i][j]+u[i][j+1])+
                    gamma*fabs(v[i][j]+v[i+1][j])*(u[i][j]-u[i][j+1])-
                    (v[i][j-1]+v[i+1][j-1])*(u[i][j-1]+u[i][j])-
                    gamma*fabs(v[i][j-1]+v[i+1][j-1])*(u[i][j-1]-u[i][j]))
                    /(4.0*dely);
                laplu = (u[i+1][j]-2.0*u[i][j]+u[i-1][j])/delx/delx+
                    (u[i][j+1]-2.0*u[i][j]+u[i][j-1])/dely/dely;
   
                f[i][j] = u[i][j]+del_t*(laplu/Re-du2dx-duvdy);
            } else {
                f[i][j] = u[i][j];
            }
        }
    }

    // Start the second parallel for loop
    // The alternative would be to run as parallel tasks as well but typically the tile is large enough to occupy all the resources available
    #pragma omp for collapse(2)
    for (i=max(1, tile_data->start_x); i<=min(tile_data->end_x-1, imax); i++) {
        for (j=max(1, tile_data->start_y); j<=min(tile_data->end_y-1, jmax-1); j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                duvdx = ((u[i][j]+u[i][j+1])*(v[i][j]+v[i+1][j])+
                    gamma*fabs(u[i][j]+u[i][j+1])*(v[i][j]-v[i+1][j])-
                    (u[i-1][j]+u[i-1][j+1])*(v[i-1][j]+v[i][j])-
                    gamma*fabs(u[i-1][j]+u[i-1][j+1])*(v[i-1][j]-v[i][j]))
                    /(4.0*delx);
                dv2dy = ((v[i][j]+v[i][j+1])*(v[i][j]+v[i][j+1])+
                    gamma*fabs(v[i][j]+v[i][j+1])*(v[i][j]-v[i][j+1])-
                    (v[i][j-1]+v[i][j])*(v[i][j-1]+v[i][j])-
                    gamma*fabs(v[i][j-1]+v[i][j])*(v[i][j-1]-v[i][j]))
                    /(4.0*dely);

                laplv = (v[i+1][j]-2.0*v[i][j]+v[i-1][j])/delx/delx+
                    (v[i][j+1]-2.0*v[i][j]+v[i][j-1])/dely/dely;

                g[i][j] = v[i][j]+del_t*(laplv/Re-duvdx-dv2dy);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }
    /* f & g at external boundaries */
    // int i,j;
    // Parallelise but typically small compared to the previous loop so they don't really make a difference
    #pragma omp for
    for (j=max(1, tile_data->start_y); j<=min(tile_data->end_y-1, jmax); j++) {
        // if (tile_data->start_x == 0) {
            f[0][j]    = u[0][j];
        // }
        // if (tile_data->end_x >= imax) {
            f[imax][j] = u[imax][j];
        // }
    }
    #pragma omp for
    for (i=max(1, tile_data->start_x); i<=min(tile_data->end_x-1, imax); i++) {
        // if (tile_data->start_y == 0) {
            g[i][0]    = v[i][0];
        // }
        // if (tile_data->end_y >= jmax) {
            g[i][jmax] = v[i][jmax];
        // }
    }
    }
    // Synchronise f and g as they are used elsewhere in a 5 stencil pattern and so need the edge data
    halo_sync(proc, f, tile_data, sync_time_taken);
    halo_sync(proc, g, tile_data, sync_time_taken);
}


/* Calculate the right hand side of the pressure equation */
void compute_rhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely, struct TileData* tile_data)
{
    int i, j;
    #pragma omp parallel for collapse(2) private(i, j) firstprivate(imax, jmax, del_t, delx, dely, tile_data, f, g, rhs, flag) default(none)
    for (i=max(1, tile_data->start_x);i<=min(imax, tile_data->end_x-1);i++) {
        for (j=max(1, tile_data->start_y);j<=min(jmax, tile_data->end_y-1);j++) {
            if (flag[i][j] & C_F) {
                /* only for fluid and non-surface cells */
                rhs[i][j] = (
                             (f[i][j]-f[i-1][j])/delx +
                             (g[i][j]-g[i][j-1])/dely
                            ) / del_t;
            }
        }
    }
    // RHS doesn't need to be synced as all RHS accesses take place within the same tile (rhs[i][j])
    // halo_sync(proc, rhs, tile_data, sync_time_taken);
}

/* Red/Black SOR to solve the poisson equation */
int poisson(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull, struct TileData* tile_data, double * sync_time_taken,
     double* possion_p_loop_time_taken, double* possion_res_loop_time_taken)
{
    int i, j, iter;
    float add, beta_2, beta_mod = 0.0;
    float p0 = 0.0;
    
    int rb; /* Red-black value. */

    float rdx2 = 1.0/(delx*delx);
    float rdy2 = 1.0/(dely*dely);
    beta_2 = -omega/(2.0*(rdx2+rdy2));

    /* Calculate sum of squares */
    #pragma omp parallel for private(i, j) firstprivate(tile_data, imax, jmax, flag, p) reduction(+:p0) default(none) collapse(2)
    for (i = max(1, tile_data->start_x); i <= min(imax, tile_data->end_x-1); i++) {
        for (j= max(1, tile_data->start_y); j<=min(jmax, tile_data->end_y-1); j++) {
            if (flag[i][j] & C_F) { p0 += p[i][j]*p[i][j]; }
        }
    }

    // Perform an all reduce sum with the local p0 sums
    // to get the real p0 to every thread
    double start_out = MPI_Wtime();
    float p0sum;
    MPI_Allreduce(&p0, &p0sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    p0 = p0sum;
    *sync_time_taken += MPI_Wtime() - start_out;

    p0 = sqrt(p0/ifull);
    if (p0 < 0.0001) { p0 = 1.0; }

    int i_start = max(1, tile_data->start_x);
    int i_end = min(imax, tile_data->end_x - 1);
    int j_start = max(1, tile_data->start_y);
    int j_end = min(jmax, tile_data->end_y - 1);
    float res_sum_local = 0.0;
    /* Red/Black SOR-iteration */
    
    // Create the parallel region here as the loop may run up to 100 times
    // The overhead of creating/joining threads 3x100 times would be extremely high
    // Local iteration variables are kept private. Iter has to have a local copy to prevent threads all incrementing the shared variable. iter is updated from each thread's local copy of iter_local
    // Iteration, matrices and timing variables are shared between threads
    // The rest are copied to a thread private copy
    #pragma omp parallel private(i, j, add, rb) shared(iter, res_sum_local, sync_time_taken, possion_p_loop_time_taken, possion_res_loop_time_taken) firstprivate(i_start, i_end, j_start, j_end,omega, beta_2, rdx2, rdy2, beta_mod, itermax, res, tile_data, proc, imax, jmax, eps, p0, ifull, nprocs, p, rhs, flag)
    {
    for (int iter_local = 0; iter_local < itermax; iter_local++) {
        double start = 0.0;
        for (rb = 0; rb <= 1; rb++) {
            start = MPI_Wtime();

            // Start the for loop with the threads already created
            // No need for private i,j as they are already thread private
            // Note: doesn't use collapse as the offset which eliminiates a branch from the hot loop body
            // depends on i
            #pragma omp for collapse(2)
            for (i = i_start; i <=i_end; i++) {
                // int offset = ((i + j_start) % 2 != rb);
                for (j = j_start; j <= j_end; j++) {
                    if ((i+j) % 2 != rb) { continue; }
                    if (flag[i][j] == (C_F | B_NSEW)) {
                        /* five point star for interior fluid cells */
                        p[i][j] = (1.-omega)*p[i][j] - 
                              beta_2*(
                                    (p[i+1][j]+p[i-1][j])*rdx2
                                  + (p[i][j+1]+p[i][j-1])*rdy2
                                  -  rhs[i][j]
                              );
                    } else if (flag[i][j] & C_F) { 
                        /* modified star near boundary */
                        beta_mod = -omega/((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2);
                        p[i][j] = (1.-omega)*p[i][j] -
                            beta_mod*(
                                  (eps_E*p[i+1][j]+eps_W*p[i-1][j])*rdx2
                                + (eps_N*p[i][j+1]+eps_S*p[i][j-1])*rdy2
                                - rhs[i][j]
                            );
                    }
                    // printf("%d: %d\n", omp_get_thread_num(), i);
                } /* end of j */
            } /* end of i */

            // only allow a single thread to run this block, blocks for the other threads
            #pragma omp single
            {   
                // Update time taken
                double time_taken = MPI_Wtime() - start;
                *possion_p_loop_time_taken += time_taken;
//                printf("%d: loop %f\n", omp_get_thread_num(), MPI_Wtime() - start);
                // Only a single thread is allowed to sync the matrix
                halo_sync(proc, p, tile_data, sync_time_taken);
                // Ensure only a single thread sets this to 0.0
                res_sum_local = 0.0;
            }
        } /* end of rb */

        start = MPI_Wtime();
        // Start a parallel for reduction using the res_sum_local variable
        #pragma omp for reduction(+:res_sum_local) collapse(2)
        for (i = i_start; i <= i_end; i++) {
            for (j = j_start; j <= j_end; j++) {
                if (flag[i][j] & C_F) {
                    /* only fluid cells */
                    add = (eps_E*(p[i+1][j]-p[i][j]) - 
                        eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
                        (eps_N*(p[i][j+1]-p[i][j]) -
                        eps_S*(p[i][j]-p[i][j-1])) * rdy2  -  rhs[i][j];
                    res_sum_local += add*add;
                }
            }
        }
        double time_taken = MPI_Wtime() - start;
//        printf("%f res sum local\n", MPI_Wtime() - start);

        start = MPI_Wtime();
        // sum up the residual across MPI processes
        // Only a single thread can execute the MPI send/recv 
        #pragma omp single
        {   
            // Update time taken
            *possion_res_loop_time_taken += time_taken;
            /* Partial computation of residual */
            *res = res_sum_local;
            // Perform a sum reduction and send result to all processes
            MPI_Allreduce(&res_sum_local, res, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            *res = sqrt((*res)/ifull)/p0;
            // Update the iterator
            iter = iter_local;
        }
//        printf("%f res bcast\n", MPI_Wtime() - start);
        /* convergence? */
        if (*res<eps) break;
    } /* end of iter */
    }
    return iter;
}


/* Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void update_velocity(float **u, float **v, float **f, float **g, float **p,
    char **flag, int imax, int jmax, float del_t, float delx, float dely, struct TileData* tile_data, double * sync_time_taken)
{   
    #pragma omp parallel firstprivate(u, v, f, g, p, flag, imax, jmax, del_t, delx, dely, tile_data) default(none)
    {
        int i, j;
        #pragma omp for collapse(2)
        for (i=max(1, tile_data->start_x); i<=min(imax-1, tile_data->end_x-1); i++) {
            for (j=max(1, tile_data->start_y); j<=min(jmax, tile_data->end_y-1); j++) {
                /* only if both adjacent cells are fluid cells */
                if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                    u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
                }
            }
        }
        #pragma omp for collapse(2)
        for (i=max(1, tile_data->start_x); i<=min(imax, tile_data->end_x - 1); i++) {
            for (j=max(1, tile_data->start_y); j<=min(jmax-1, tile_data->end_y - 1); j++) {
                /* only if both adjacent cells are fluid cells */
                if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                    v[i][j] = g[i][j]-(p[i][j+1]-p[i][j])*del_t/dely;
                }
            }
        }
    }
    // Sync u,v as they have been updated and are accessed elsewhere in a 5 stencil pattern
    halo_sync(proc, u, tile_data, sync_time_taken);
    halo_sync(proc, v, tile_data, sync_time_taken);
}


/* Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions (ie no particle moves more than one cell width in one
 * timestep). Otherwise the simulation becomes unstable.
 */
void set_timestep_interval(float *del_t, int imax, int jmax, float delx,
    float dely, float **u, float **v, float Re, float tau, struct TileData* tile_data, double * sync_time_taken)
{
    int i, j;
    float umax, vmax, umax_local, vmax_local, deltu, deltv, deltRe; 

    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10) { /* else no time stepsize control */
        umax_local = 1.0e-10;
        vmax_local = 1.0e-10; 
        // No parallelisation as the overhead is too high for gains possible
        // Calculate the local umax and vmax

        #pragma omp parallel private(i, j) firstprivate(imax, jmax, tile_data, u, v) shared(umax_local, vmax_local) default(none)
        {
            #pragma omp for collapse(2) reduction(max:umax_local)
            for (i=tile_data->start_x; i<=min(imax+1, tile_data->end_x - 1); i++) {
                for (j=max(1, tile_data->start_y); j<=min(jmax+1, tile_data->end_y - 1); j++) {
                    umax_local = max(fabs(u[i][j]), umax_local);
                }
            }
            #pragma omp for collapse(2) reduction(max:vmax_local)
            for (i=max(1, tile_data->start_x); i<=min(imax+1, tile_data->end_x - 1); i++) {
                for (j=tile_data->start_y; j<=min(jmax+1, tile_data->end_y - 1); j++) {
                    vmax_local = max(fabs(v[i][j]), vmax_local);
                }
            }
        }

        // calculate the global umax and vmax by performing a max reduction on both vars
        // using MPI_Allreduce
        double start = MPI_Wtime();
        float max_buffer[2];
        max_buffer[0] = umax_local;
        max_buffer[1] = vmax_local;
        float max_buffer2[2];
        MPI_Allreduce(&max_buffer, &max_buffer2, 2, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        umax = max_buffer2[0];
        vmax = max_buffer2[1];
        *sync_time_taken += MPI_Wtime() - start;

        deltu = delx/umax;
        deltv = dely/vmax;
        deltRe = 1/(1/(delx*delx)+1/(dely*dely))*Re/2.0;

        if (deltu<deltv) {
            *del_t = min(deltu, deltRe);
        } else {
            *del_t = min(deltv, deltRe);
        }
        *del_t = tau * (*del_t); /* multiply by safety factor */
    }
}
