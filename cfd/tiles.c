#define t_start 0.0001
#define t_byte 0.0001
#include <stdio.h>
#include <mpi.h>
#include "tiles.h"

int min(int a, int b);

void init_tile_data(int rank, int nprocs, int mesh_width, int mesh_height, struct TileData* tile_data) {
	init_tile_shape(nprocs, mesh_width, mesh_height, tile_data);
	init_tile_pos(rank, tile_data);
	init_tile_start_end(mesh_width, mesh_height, tile_data);
	init_tile_datatypes(tile_data);
}

void init_tile_shape(int nprocs, int mesh_width, int mesh_height, struct TileData* tile_data) {
	float min_comm_cost = 1e10;
	int tiles_num_x, tiles_num_y;
	for (tiles_num_x = 1; tiles_num_x <= nprocs; tiles_num_x++) {
		for (tiles_num_y = 1; tiles_num_y <= nprocs; tiles_num_y++) {
			if (tiles_num_x * tiles_num_y == nprocs) {
				// All procs are used
				int tile_width_current = mesh_width / tiles_num_x;
				int tile_height_current = mesh_height / tiles_num_y;

				if (tile_width_current * tiles_num_x < mesh_width) {
					tile_width_current += 1;
				}
				if (tile_height_current * tiles_num_y < mesh_height) {
					tile_height_current += 1;
				}

				float comm_cost_x = 2 * (t_start + t_byte * tile_height_current);
				float comm_cost_y = 2 * (t_start + t_byte * tile_width_current);

				if (tiles_num_x == 1) {
					comm_cost_x = 0.0;
				}
				if (tiles_num_y == 1) {
					comm_cost_y = 0.0;
				}

				float comm_cost = comm_cost_x + comm_cost_y;
				// printf("%d %d %d %f\n", nprocs, tiles_num_x, tiles_num_y, comm_cost);

				if (comm_cost < min_comm_cost) {
					min_comm_cost = comm_cost;
					tile_data->num_x = tiles_num_x;
					tile_data->num_y = tiles_num_y;
					tile_data->width = tile_width_current;
					tile_data->height = tile_height_current;
				}
			}
		}	
	}
}

void init_tile_pos(int rank, struct TileData* tile_data) {
	tile_data->pos_x = rank % tile_data->num_x;
	tile_data->pos_y = rank / tile_data->num_x;
}

void init_tile_start_end(int mesh_width, int mesh_height, struct TileData* tile_data) {
	tile_data->start_x = tile_data->pos_x * tile_data->width;
	tile_data->start_y = tile_data->pos_y * tile_data->height;
	tile_data->end_x = min(tile_data->start_x + tile_data->width, mesh_width);
	tile_data->end_y = min(tile_data->start_y + tile_data->height, mesh_height);
}

void init_tile_datatypes(struct TileData* tile_data) {
	MPI_Type_contiguous(tile_data->height, MPI_FLOAT, &tile_data->tilecoltype);
    MPI_Type_commit(&tile_data->tilecoltype);
	MPI_Type_vector(tile_data->width, 1, tile_data->height, MPI_FLOAT, &tile_data->tilerowtype);
    MPI_Type_commit(&tile_data->tilerowtype);
}

void free_tile_data(struct TileData* tile_data) {
	MPI_Type_free(&tile_data->tilecoltype);
    MPI_Type_free(&tile_data->tilerowtype);
}


void halo_sync(int rank, float **array, struct TileData* tile_data) {
	// MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
	MPI_Request requests[8];
	int requests_pos = 0;
	// No -1 after end_? as the range is non inclusive, so end_? is from the next tile.
	// printf("Rank %d checking in\n", rank);
	// Receive the data asynchronously to avoid a deadlock and improve performance
	if (tile_data->pos_x > 0) {
		// There is a tile to the left
		MPI_Irecv(&array[tile_data->start_x-1][tile_data->start_y], 1, tile_data->tilecoltype, rank - 1, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		// printf("I am %d, getting from %d\n", rank, rank - 1);
		requests_pos++;
	}
	if (tile_data->pos_x < tile_data->num_x - 1) {
		// There is a tile to the right
		MPI_Irecv(&array[tile_data->end_x][tile_data->start_y], 1, tile_data->tilecoltype, rank + 1, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		// printf("I am %d, getting from %d. x: %d, y:%d\n", rank, rank + 1, tile_data->end_x, tile_data->start_y);
		requests_pos++;
	}
	if (tile_data->pos_y > 0) {
		// There is a tile above
		int target_rank = rank - tile_data->num_x;
		MPI_Irecv(&array[tile_data->start_x][tile_data->start_y - 1], 1, tile_data->tilerowtype, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		// printf("I am %d, getting from %d\n", rank, target_rank);
		requests_pos++;
	}
	if (tile_data->pos_y < tile_data->num_y - 1) {
		// There is a tile below
		int target_rank = rank + tile_data->num_x;
		MPI_Irecv(&array[tile_data->start_x][tile_data->end_y], 1, tile_data->tilerowtype, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		// printf("I am %d, getting from %d\n", rank, target_rank);
		requests_pos++;
	}

	// Send the data asynchronously
	if (tile_data->pos_x > 0) {
		// There is a tile to the left
		MPI_Isend(&array[tile_data->start_x][tile_data->start_y], 1, tile_data->tilecoltype, rank - 1, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		// printf("I am %d, sending to %d\n", rank, rank - 1);
		requests_pos++;
	}
	if (tile_data->pos_x < tile_data->num_x - 1) {
		// There is a tile to the right
		MPI_Isend(&array[tile_data->end_x - 1][tile_data->start_y], 1, tile_data->tilecoltype, rank + 1, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		// printf("I am %d, sending to %d\n", rank, rank + 1);
		requests_pos++;
	}
	if (tile_data->pos_y > 0) {
		// There is a tile above
		int target_rank = rank - tile_data->num_x;
		MPI_Isend(&array[tile_data->start_x][tile_data->start_y], 1, tile_data->tilerowtype, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		// printf("I am %d, sending to %d\n", rank, target_rank);
		requests_pos++;
	}
	if (tile_data->pos_y < tile_data->num_y - 1) {
		// There is a tile below
		int target_rank = rank + tile_data->num_x;
		MPI_Isend(&array[tile_data->start_x][tile_data->end_y - 1], 1, tile_data->tilerowtype, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		// printf("I am %d, sending to %d\n", rank, target_rank);
		requests_pos++;
	}

	MPI_Status s;
	for (int i = 0; i < requests_pos; i++) {
		// Need to wait for all receives and sends to complete before reading or writing data from the array
		MPI_Wait(&requests[i], &s);
		// TODO check s is ok
	}
	printf("Sync for %d took %f seconds\n", rank, MPI_Wtime() - start);
	// MPI_Barrier(MPI_COMM_WORLD);
}

int min(int a, int b) {
	return (a > b) ? b : a;
}