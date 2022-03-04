#ifndef __tiles_h_
#define __tiles_h_
#include <mpi.h>

struct TileData {
	int num_x;
	int num_y;
	int width;
	int height;
	int std_width;
	int std_height;
	int pos_x;
	int pos_y;
	int start_x;
	int start_y;
	int end_x;
	int end_y;
	MPI_Datatype tilecoltype;
	MPI_Datatype tilerowtype;
	int mesh_height;
	int mesh_width;
};

void init_tile_data(int rank, int nprocs, int mesh_width, int mesh_height, struct TileData* tile_data);
void init_tile_shape(int nprocs, int mesh_width, int mesh_height, struct TileData* tile_data);
void init_tile_pos(int rank, struct TileData* tile_data);
void init_tile_start_end(struct TileData* tile_data);
void init_tile_datatypes(struct TileData* tile_data);
void free_tile_data(struct TileData* tile_data);

void halo_sync(int rank, float **array, struct TileData* tile_data, double * sync_time_taken);
void sync_tile_to_root(int rank, float** array, struct TileData* tile_data);
void screw_it_sync_everything(int rank, float** array, struct TileData* tile_data);
void print_tile(float** array, struct TileData* tile_data);
void test_halo_sync(int rank, int nprocs);
#endif