#include "tiles.h"
void apply_boundary_conditions(float **u, float **v, char **flag,
    int imax, int jmax, float ui, float vi);
void apply_tile_boundary_conditions(float **u, float **v, char **flag,
    int imax, int jmax, float ui, float vi, struct TileData* tile_data);