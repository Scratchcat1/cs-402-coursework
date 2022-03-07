#ifndef __simulation_par_h_
#define __simulation_par_h_
#include "tiles.h"
void compute_tentative_velocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re, struct TileData* tile_data, double * sync_time_taken);

void compute_rhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely, struct TileData* tile_data);

int poisson(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull, struct TileData* tile_data, double* sync_time_taken, double* possion_p_loop_time_taken, double* possion_res_loop_time_taken);

void update_velocity(float **u, float **v, float **f, float **g, float **p,
    char **flag, int imax, int jmax, float del_t, float delx, float dely, struct TileData* tile_data, double * sync_time_taken);

void set_timestep_interval(float *del_t, int imax, int jmax, float delx,
    float dely, float **u, float **v, float Re, float tau, struct TileData* tile_data, double * sync_time_taken);
#endif