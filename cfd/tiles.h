#include <mpi.h>

struct TileData {
	int num_x;
	int num_y;
	int width;
	int height;
	int pos_x;
	int pos_y;
	int start_x;
	int start_y;
	int end_x;
	int end_y;
	MPI_Datatype tilecoltype;
	MPI_Datatype tilerowtype;
};

void init_tile_data(int rank, int nprocs, int mesh_width, int mesh_height, struct TileData* tile_data);
void init_tile_shape(int nprocs, int mesh_width, int mesh_height, struct TileData* tile_data);
void init_tile_pos(int rank, struct TileData* tile_data);
void init_tile_start_end(int mesh_width, int mesh_height, struct TileData* tile_data);
void init_tile_datatypes(struct TileData* tile_data);
void free_tile_data(struct TileData* tile_data);

void halo_sync(int rank, float **array, struct TileData* tile_data);