#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "datadef.h"
#include "init.h"
#include "tiles.h"
#include <omp.h>
#include <mpi.h>
#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

extern int *ileft, *iright;
extern int nprocs, proc;

/* Computation of tentative velocity field (f, g) */
void compute_tentative_velocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re, struct TileData* tile_data)
{
    #pragma omp parallel firstprivate(imax, jmax, del_t, delx, dely, gamma, Re)
    {
    int  i, j;
    float du2dx, duvdy, duvdx, dv2dy, laplu, laplv;
    #pragma omp for
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
    #pragma omp for
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

    for (j=max(1, tile_data->start_y); j<=min(tile_data->end_y-1, jmax); j++) {
        if (tile_data->start_x == 0) {
            f[0][j]    = u[0][j];
        }
        if (tile_data->end_x >= imax) {
            f[imax][j] = u[imax][j];
        }
    }

    for (i=max(1, tile_data->start_x); i<=min(tile_data->end_x-1, imax); i++) {
        if (tile_data->start_y == 0) {
            g[i][0]    = v[i][0];
        }
        if (tile_data->end_y >= jmax) {
            g[i][jmax] = v[i][jmax];
        }
    }
    }
    halo_sync(proc, f, tile_data);
    halo_sync(proc, g, tile_data);
}


/* Calculate the right hand side of the pressure equation */
void compute_rhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely, struct TileData* tile_data)
{
    int i, j;
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
    halo_sync(proc, rhs, tile_data);
}

/* Red/Black SOR to solve the poisson equation */
int poisson(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull, struct TileData* tile_data)
{
    int i, j, iter;
    float add, beta_2, beta_mod;
    float p0 = 0.0;
    
    int rb; /* Red-black value. */

    float rdx2 = 1.0/(delx*delx);
    float rdy2 = 1.0/(dely*dely);
    beta_2 = -omega/(2.0*(rdx2+rdy2));
    double start_out = MPI_Wtime();

    /* Calculate sum of squares */
    for (i = max(1, tile_data->start_x); i <= min(imax, tile_data->end_x-1); i++) {
        for (j= max(1, tile_data->start_y); j<=min(jmax, tile_data->end_y-1); j++) {
            if (flag[i][j] & C_F) { p0 += p[i][j]*p[i][j]; }
        }
    }
//    printf("%f sum of squares\n", MPI_Wtime() - start_out);
//    start_out = MPI_Wtime();
//    float* recv_buffer = NULL;
 //   if (proc == 0) {
  //      recv_buffer = malloc(sizeof(float) * nprocs);
  // }
//    MPI_Gather(&p0, 1, MPI_FLOAT, recv_buffer, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
//    if (proc == 0) {
//        float p0sum = 0.0;
//        for (i = 0; i < nprocs; i++) {
//            p0sum += recv_buffer[i];
//        }
//        p0 = p0sum;
//        free(recv_buffer);
//    }
//    MPI_Bcast(&p0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    float p0sum;
    MPI_Allreduce(&p0, &p0sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    p0 = p0sum;
//    printf("num threads %d\n", omp_get_max_threads());
//    printf("%f p0 sync\n", MPI_Wtime() - start_out);

    p0 = sqrt(p0/ifull);
    if (p0 < 0.0001) { p0 = 1.0; }

    int i_start = max(1, tile_data->start_x);
    int i_end = min(imax, tile_data->end_x - 1);
    int j_start = max(1, tile_data->start_y);
    int j_end = min(jmax, tile_data->end_y - 1);
    float res_sum_local = 0.0;
    /* Red/Black SOR-iteration */
 //   printf("Going parallel\n");
    #pragma omp parallel private(i, j, add, rb) shared(iter, p, rhs, flag, res_sum_local) firstprivate(i_start, i_end, j_start, j_end,omega, beta_2, rdx2, rdy2, beta_mod, itermax, res, tile_data, proc, imax, jmax, eps, p0, ifull, nprocs)
    {
    for (int iter_local = 0; iter_local < itermax; iter_local++) {
        double start = 0.0;
        for (rb = 0; rb <= 1; rb++) {
            start = MPI_Wtime();
            #pragma omp for  //private(i, j) //firstprivate(rb, i_start, i_end, j_start, j_end,omega, beta_2, rdx2, rdy2, beta_mod)
            for (i = i_start; i <=i_end; i++) {
                int offset = ((i + j_start) % 2 != rb);
                for (j = j_start+offset; j <= j_end; j += 2) {
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
         //   #pragma omp barrier
            #pragma omp single
            {
//                printf("%d: loop %f\n", omp_get_thread_num(), MPI_Wtime() - start);
                halo_sync(proc, p, tile_data);
                res_sum_local = 0.0;
            }
        } /* end of rb */

        start = MPI_Wtime();
        #pragma omp for reduction(+:res_sum_local)
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
//        printf("%f res sum local\n", MPI_Wtime() - start);
        start = MPI_Wtime();
        #pragma omp single
        {
            /* Partial computation of residual */
            *res = res_sum_local;
            MPI_Allreduce(&res_sum_local, res, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            *res = sqrt((*res)/ifull)/p0;
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
    char **flag, int imax, int jmax, float del_t, float delx, float dely, struct TileData* tile_data)
{
    int i, j;
    for (i=max(1, tile_data->start_x); i<=min(imax-1, tile_data->end_x-1); i++) {
        for (j=max(1, tile_data->start_y); j<=min(jmax, tile_data->end_y-1); j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
            }
        }
    }
    for (i=max(1, tile_data->start_x); i<=min(imax, tile_data->end_x - 1); i++) {
        for (j=max(1, tile_data->start_y); j<=min(jmax-1, tile_data->end_y - 1); j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j]-(p[i][j+1]-p[i][j])*del_t/dely;
            }
        }
    }
    halo_sync(proc, u, tile_data);
    halo_sync(proc, v, tile_data);
}


/* Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions (ie no particle moves more than one cell width in one
 * timestep). Otherwise the simulation becomes unstable.
 */
void set_timestep_interval(float *del_t, int imax, int jmax, float delx,
    float dely, float **u, float **v, float Re, float tau, struct TileData* tile_data)
{
    int i, j;
    float umax, vmax, umax_local, vmax_local, deltu, deltv, deltRe; 

    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10) { /* else no time stepsize control */
        umax_local = 1.0e-10;
        vmax_local = 1.0e-10; 
        for (i=tile_data->start_x; i<=min(imax+1, tile_data->end_x - 1); i++) {
            for (j=max(1, tile_data->start_y); j<=min(jmax+1, tile_data->end_y - 1); j++) {
                umax_local = max(fabs(u[i][j]), umax_local);
            }
        }
        for (i=max(1, tile_data->start_x); i<=min(imax+1, tile_data->end_x - 1); i++) {
            for (j=tile_data->start_y; j<=min(jmax+1, tile_data->end_y - 1); j++) {
                vmax_local = max(fabs(v[i][j]), vmax_local);
            }
        }

        float max_buffer[2];
        max_buffer[0] = umax_local;
        max_buffer[1] = vmax_local;
        float max_buffer2[2];
        MPI_Allreduce(&max_buffer, &max_buffer2, 2, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        umax = max_buffer2[0];
        vmax = max_buffer2[1];

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
