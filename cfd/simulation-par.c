#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "datadef.h"
#include "init.h"
#include "tiles.h"

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

extern int *ileft, *iright;
extern int nprocs, proc;

/* Computation of tentative velocity field (f, g) */
void compute_tentative_velocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re, struct TileData* tile_data)
{
    int  i, j;
    float du2dx, duvdy, duvdx, dv2dy, laplu, laplv;

    for (i=max(1, tile_data->start_x); i<=min(tile_data->end_x,imax-1); i++) { // i=1 i <=imax -1
        for (j=max(1, tile_data->start_y); j<=min(tile_data->end_y, jmax); j++) { // j=1 j <=jmax
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

    for (i=max(1, tile_data->start_x); i<=min(tile_data->end_x, imax); i++) {
        for (j=max(1, tile_data->start_y); j<=min(tile_data->end_y, jmax-1); j++) {
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
    for (j=max(1, tile_data->start_y); j<=min(tile_data->end_y, jmax); j++) {
        f[0][j]    = u[0][j];
        f[imax][j] = u[imax][j];
    }
    for (i=max(1, tile_data->start_x); i<=min(tile_data->end_x, imax); i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
    // if (proc == 0) {
    //     print_tile(f, tile_data);
    // }
    halo_sync(proc, f, tile_data);
    halo_sync(proc, g, tile_data);
}


/* Calculate the right hand side of the pressure equation */
void compute_rhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely, struct TileData* tile_data)
{
    int i, j;
    for (i=max(1, tile_data->start_x);i<=min(imax, tile_data->end_x);i++) {
        for (j=max(1, tile_data->start_y);j<=min(jmax, tile_data->end_y);j++) {
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

    /* Calculate sum of squares */
    for (i = max(1, tile_data->start_x); i <= min(imax, tile_data->end_x-1); i++) {
        for (j= max(1, tile_data->start_y); j<=min(jmax, tile_data->end_y-1); j++) {
            if (flag[i][j] & C_F) { p0 += p[i][j]*p[i][j]; }
        }
    }
    // printf("p0 %f\n", p0);

    float* recv_buffer = NULL;
    if (proc == 0) {
        recv_buffer = malloc(sizeof(float) * nprocs);
    }
    MPI_Gather(&p0, 1, MPI_FLOAT, recv_buffer, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (proc == 0) {
        float p0sum = 0.0;
        for (i = 0; i < nprocs; i++) {
            p0sum += recv_buffer[i];
        }
        p0 = p0sum;
        free(recv_buffer);
        // printf("sump0 %f\n", p0);
    }
    MPI_Bcast(&p0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   
    p0 = sqrt(p0/ifull);
    if (p0 < 0.0001) { p0 = 1.0; }

    /* Red/Black SOR-iteration */
    for (iter = 0; iter < itermax; iter++) {
        for (rb = 0; rb <= 1; rb++) {
            for (i = max(1, tile_data->start_x); i <= min(imax, tile_data->end_x); i++) {
                int j_start = max(1, tile_data->start_y);
                for (j = j_start; j <= min(jmax, tile_data->end_y); j += 1) {
                    if ((i+j) % 2 != rb) { continue; } // TODO Remove this branch again
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
                } /* end of j */
            } /* end of i */
            halo_sync(proc, p, tile_data);
        } /* end of rb */
        
        /* Partial computation of residual */
        *res = 0.0;
        for (i = max(1, tile_data->start_x); i <= min(imax, tile_data->end_x - 1); i++) {
            for (j = max(1, tile_data->start_y); j <= min(jmax, tile_data->end_y - 1); j++) {
                if (flag[i][j] & C_F) {
                    /* only fluid cells */
                    add = (eps_E*(p[i+1][j]-p[i][j]) - 
                        eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
                        (eps_N*(p[i][j+1]-p[i][j]) -
                        eps_S*(p[i][j]-p[i][j-1])) * rdy2  -  rhs[i][j];
                    *res += add*add;
                }
            }
        }

        recv_buffer = NULL;
        if (proc == 0) {
            recv_buffer = malloc(sizeof(float) * nprocs);
        }
        MPI_Gather(res, 1, MPI_FLOAT, recv_buffer, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (proc == 0) {
            float res_sum = 0.0;
            for (i = 0; i < nprocs; i++) {
                res_sum += recv_buffer[i];
            }
            free(recv_buffer);
            *res = res_sum;
            *res = sqrt((*res)/ifull)/p0;
            // printf("res %f p0 %f\n", *res, p0);
        }
        MPI_Bcast(res, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        /* convergence? */
        if (*res<eps) break;
    } /* end of iter */

    return iter;
}


/* Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void update_velocity(float **u, float **v, float **f, float **g, float **p,
    char **flag, int imax, int jmax, float del_t, float delx, float dely, struct TileData* tile_data)
{
    int i, j;

    for (i=max(1, tile_data->start_x); i<=min(imax-1, tile_data->end_x); i++) {
        for (j=max(1, tile_data->start_y); j<=min(jmax, tile_data->end_y); j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
            }
        }
    }
    for (i=max(1, tile_data->start_x); i<=min(imax, tile_data->end_x); i++) {
        for (j=max(1, tile_data->start_y); j<=min(jmax-1, tile_data->end_y); j++) {
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
        float* recv_buffer = NULL;
        if (proc == 0) {
            recv_buffer = malloc(sizeof(float) * 2 * nprocs);
        }
        MPI_Gather(&max_buffer, 2, MPI_FLOAT, recv_buffer, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (proc == 0) {
            umax = recv_buffer[0];
            vmax = recv_buffer[1];
            for (i = 0; i < nprocs * 2; i += 2) {
                umax = max(umax, recv_buffer[i]);
                vmax = max(vmax, recv_buffer[i + 1]);
            }
            max_buffer[0] = umax;
            max_buffer[1] = vmax;
        }
        MPI_Bcast(&max_buffer, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
        umax = max_buffer[0];
        vmax = max_buffer[1];

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
    printf("delt %f\n", *del_t);
}
