#define t_start 0.00005
#define t_byte 0.0000001
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include "tiles.h"
#include "alloc.h"

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

void init_tile_data(int rank, int nprocs, int mesh_width, int mesh_height, struct TileData* tile_data) {
	// Init the tile data, calculating shape, position, start and ends and the datatypes
	init_tile_shape(nprocs, mesh_width, mesh_height, tile_data);
	init_tile_pos(rank, tile_data);
	init_tile_start_end(tile_data);
	init_tile_datatypes(tile_data);
}

void init_tile_shape(int nprocs, int mesh_width, int mesh_height, struct TileData* tile_data) {
	// This function finds the minimum communication cost layout
	float min_comm_cost = 1e10;
	int tiles_num_x, tiles_num_y;
	// Test each possible tile shape where all the processors are used
	for (tiles_num_x = 1; tiles_num_x <= nprocs; tiles_num_x++) {
		for (tiles_num_y = 1; tiles_num_y <= nprocs; tiles_num_y++) {
			// Only accept valid shapes
			if (tiles_num_x * tiles_num_y == nprocs) {
				// Calculate the width of each tile (integer division so tile_width_current * num_x may be < mesh_width)
				int tile_width_current = mesh_width / tiles_num_x;
				int tile_height_current = mesh_height / tiles_num_y;

				// Will fail if the number of procs in the axis is > 1/2 axis length. It is reasonable that this will not take place.
				// Correct the with to ensure the entire mesh is covered
				if (tile_width_current * tiles_num_x < mesh_width) {
					tile_width_current += 1;
				}
				if (tile_height_current * tiles_num_y < mesh_height) {
					tile_height_current += 1;
				}

				// Estimate the cost of communication for each axis
				// Assumes a middle node transfering left/right and up/down
				// as that will be the slowest to transfer
				// and the other processes will have to wait
				float comm_cost_x = 2 * (t_start + t_byte * tile_height_current);
				float comm_cost_y = 2 * (t_start + t_byte * tile_width_current);
				
				// Ignore the cost if no transfers occur in that axis
				if (tiles_num_x == 1) {
					comm_cost_x = 0.0;
				}
				if (tiles_num_y == 1) {
					comm_cost_y = 0.0;
				}

				float comm_cost = comm_cost_x + comm_cost_y;
				// printf("%d %d %d %f\n", nprocs, tiles_num_x, tiles_num_y, comm_cost);
				
				// If this is a smaller cost use it
				if (comm_cost < min_comm_cost) {
					min_comm_cost = comm_cost;
					tile_data->num_x = tiles_num_x;
					tile_data->num_y = tiles_num_y;
					tile_data->width = tile_width_current;
					tile_data->height = tile_height_current;
					tile_data->std_width = tile_width_current;
					tile_data->std_height = tile_height_current;
				}
			}
		}	
	}
	tile_data->mesh_height = mesh_height;
	tile_data->mesh_width = mesh_width;
}

void init_tile_pos(int rank, struct TileData* tile_data) {
	// Calculate the position in x and y from the process rank
	// results in a layout like this ( x across, y down)
	// 0  1  2  3
	// 4  5  6  7
	// 8  9  10 11
	// 12 13 14 15
	tile_data->pos_x = rank % tile_data->num_x;
	tile_data->pos_y = rank / tile_data->num_x;
}

void init_tile_start_end(struct TileData* tile_data) {
	// Calculate the start and end of the tile
	tile_data->start_x = tile_data->pos_x * tile_data->width;
	tile_data->start_y = tile_data->pos_y * tile_data->height;
	// Ensure tile doesn't overrun the end of the mesh
	tile_data->end_x = min(tile_data->start_x + tile_data->width, tile_data->mesh_width);
	tile_data->end_y = min(tile_data->start_y + tile_data->height, tile_data->mesh_height);

	// Ensure the width is consistent with start + end
	// Important for edge tiles which may be smaller than the rest
	tile_data->width = tile_data->end_x - tile_data->start_x;
	tile_data->height = tile_data->end_y - tile_data->start_y;
}

void init_tile_datatypes(struct TileData* tile_data) {
	// Allocate the datatypes 
	// Columns are contigious so allocate a region the length of the tile height
	MPI_Type_contiguous(tile_data->height, MPI_FLOAT, &tile_data->tilecoltype);
    MPI_Type_commit(&tile_data->tilecoltype);
	// Rows are the width in length with mesh height values between them.
	MPI_Type_vector(tile_data->width, 1, tile_data->mesh_height, MPI_FLOAT, &tile_data->tilerowtype);
    MPI_Type_commit(&tile_data->tilerowtype);
}

void free_tile_data(struct TileData* tile_data) {
	// Free the types
	MPI_Type_free(&tile_data->tilecoltype);
    MPI_Type_free(&tile_data->tilerowtype);
}


void halo_sync(int rank, float** array, struct TileData* tile_data, double * sync_time_taken) {
	// Perform a halosync for each edge and the corners (compute velocity is not a 5 stencil)
	double start = MPI_Wtime();

	// Keep track of the requests in an array
	MPI_Request requests[16];
	int requests_pos = 0;
	// No -1 after end_? as the range is non inclusive, so end_? is from the next tile.
//      MPI_Barrier(MPI_COMM_WORLD);

	// Flags to check if the tile should sync in certain directions
	int sync_left = tile_data->pos_x > 0;
	int sync_right = tile_data->pos_x < tile_data->num_x - 1;
	int sync_up = tile_data->pos_y > 0;
	int sync_down = tile_data->pos_y < tile_data->num_y - 1;

	// Receive the data asynchronously to avoid a deadlock and improve performance
	if (sync_left) {
		// There is a tile to the left
		MPI_Irecv(&array[tile_data->start_x-1][tile_data->start_y], 1, tile_data->tilecoltype, rank - 1, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_right) {
		// There is a tile to the right
		MPI_Irecv(&array[tile_data->end_x][tile_data->start_y], 1, tile_data->tilecoltype, rank + 1, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_up) {
		// There is a tile above
		int target_rank = rank - tile_data->num_x;
		MPI_Irecv(&array[tile_data->start_x][tile_data->start_y - 1], 1, tile_data->tilerowtype, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_down) {
		// There is a tile below
		int target_rank = rank + tile_data->num_x;
		MPI_Irecv(&array[tile_data->start_x][tile_data->end_y], 1, tile_data->tilerowtype, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	
	// Diagonals, single float values
	if (sync_left && sync_up) {
		int target_rank = rank - tile_data->num_x - 1;
		MPI_Irecv(&array[tile_data->start_x-1][tile_data->start_y - 1], 1, MPI_FLOAT, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_up && sync_right) {
		int target_rank = rank - tile_data->num_x + 1;
		MPI_Irecv(&array[tile_data->end_x][tile_data->start_y - 1], 1, MPI_FLOAT, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_right && sync_down) {
		int target_rank = rank + tile_data->num_x + 1;
		MPI_Irecv(&array[tile_data->end_x][tile_data->end_y], 1, MPI_FLOAT, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_down && sync_left) {
		int target_rank = rank + tile_data->num_x - 1;
		MPI_Irecv(&array[tile_data->start_x-1][tile_data->end_y], 1, MPI_FLOAT, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}

	// Send the data asynchronously
	if (sync_left) {
		// There is a tile to the left
		MPI_Isend(&array[tile_data->start_x][tile_data->start_y], 1, tile_data->tilecoltype, rank - 1, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_right) {
		// There is a tile to the right
		MPI_Isend(&array[tile_data->end_x - 1][tile_data->start_y], 1, tile_data->tilecoltype, rank + 1, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_up) {
		// There is a tile above
		int target_rank = rank - tile_data->num_x;
		MPI_Isend(&array[tile_data->start_x][tile_data->start_y], 1, tile_data->tilerowtype, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_down) {
		// There is a tile below
		int target_rank = rank + tile_data->num_x;
		MPI_Isend(&array[tile_data->start_x][tile_data->end_y - 1], 1, tile_data->tilerowtype, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}

	// Diagonals, send single floats
	if (sync_left && sync_up) {
		int target_rank = rank - tile_data->num_x - 1;
		MPI_Isend(&array[tile_data->start_x][tile_data->start_y], 1, MPI_FLOAT, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_up && sync_right) {
		int target_rank = rank - tile_data->num_x + 1;
		MPI_Isend(&array[tile_data->end_x - 1][tile_data->start_y], 1, MPI_FLOAT, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_right && sync_down) {
		int target_rank = rank + tile_data->num_x + 1;
		MPI_Isend(&array[tile_data->end_x - 1][tile_data->end_y - 1], 1, MPI_FLOAT, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}
	if (sync_down && sync_left) {
		int target_rank = rank + tile_data->num_x - 1;
		MPI_Isend(&array[tile_data->start_x][tile_data->end_y - 1], 1, MPI_FLOAT, target_rank, 0, MPI_COMM_WORLD, &requests[requests_pos]);
		requests_pos++;
	}

	MPI_Status s;
	for (int i = 0; i < requests_pos; i++) {
		// Need to wait for all receives and sends to complete before reading or writing data from the array
		MPI_Wait(&requests[i], &s);
	}
	double time_taken = MPI_Wtime() - start;
	// printf("Sync took %f seconds\n", time_taken);
	*sync_time_taken += time_taken;
}

void sync_tile_to_root(int rank, float** array, struct TileData* tile_data) {
	// Copy data from other mpi processes to process 0

	// Construct the datatypes for each tile
	MPI_Datatype full_tile, right_tile, bottom_tile, bottom_right_tile;
	// Widths may be different for tiles on the right/bottom if num proces in axis is not a multiple of the mesh size
	int right_tile_width = tile_data->std_width- (tile_data->std_width * tile_data->num_x - tile_data->mesh_width);
	int bottom_tile_height = tile_data->std_height-(tile_data->std_height * tile_data->num_y - tile_data->mesh_height);

	// A full tile is width number of columns, each a mesh height of cells apart
	MPI_Type_vector(tile_data->std_width, tile_data->std_height, tile_data->mesh_height, MPI_FLOAT, &full_tile);
    MPI_Type_commit(&full_tile);

	// The right tile may be narrower, use right_tile_width
	MPI_Type_vector(right_tile_width, tile_data->std_height, tile_data->mesh_height, MPI_FLOAT, &right_tile);
    MPI_Type_commit(&right_tile);

	// The bottom tile may be shorter, use bottom_tile_width
	MPI_Type_vector(tile_data->std_width, bottom_tile_height, tile_data->mesh_height, MPI_FLOAT, &bottom_tile);
    MPI_Type_commit(&bottom_tile);

	// Bottom right tile may be narrower and shorter
	MPI_Type_vector(right_tile_width, bottom_tile_height, tile_data->mesh_height, MPI_FLOAT, &bottom_right_tile);
    MPI_Type_commit(&bottom_right_tile);

	if (rank == 0) {
		int nprocs = tile_data->num_x * tile_data->num_y;
		// Keep track of all the async requests
		MPI_Request* requests = malloc(sizeof(MPI_Request) * nprocs);

		// Collect data from each of the other ranks
		for (int other_rank = 1; other_rank < nprocs; other_rank++) {
			// Calculate the position of the other rank
			int other_pos_x = other_rank % tile_data->num_x;
			int other_pos_y = other_rank / tile_data->num_x;

			// Select the right datatype for the other rank tile
			MPI_Datatype* dtype;
			int on_right = other_pos_x == tile_data->num_x - 1;
			int on_bottom = other_pos_y == tile_data->num_y - 1;
			if (on_right && on_bottom) {
				dtype = &bottom_right_tile;
			} else if (on_right) {
				dtype = &right_tile;
			} else if (on_bottom) {
				dtype = &bottom_tile;
			} else {
				dtype = &full_tile;
			}
			// Calculate where the tile starts
			int other_start_x = other_pos_x * tile_data->std_width;
			int other_start_y = other_pos_y * tile_data->std_height;

			// Receive async as there will be many tiles trying to send data
			MPI_Irecv(&array[other_start_x][other_start_y], 1, *dtype, other_rank, 0, MPI_COMM_WORLD, &requests[other_rank]);
		}

		// Ensure all requests complete before continuing
		MPI_Status s;
		for (int i = 1; i < nprocs; i++) {
			MPI_Wait(&requests[i], &s);
		}
		printf("Recv ok\n");
		free(requests);
		// Async receive
	} else {
		// Async send
		MPI_Datatype* dtype = &full_tile;
		int on_right = tile_data->pos_x == tile_data->num_x - 1;
		int on_bottom = tile_data->pos_y == tile_data->num_y - 1;
		// Select the right dtype
		if (on_right && on_bottom) {
			dtype = &bottom_right_tile;
		} else if (on_right) {
			dtype = &right_tile;
		} else if (on_bottom) {
			dtype = &bottom_tile;
		}
		// Send synchronously as this is the only transfer required
		MPI_Send(&array[tile_data->start_x][tile_data->start_y], 1, *dtype, 0, 0, MPI_COMM_WORLD);
	}

	// Free the tiles afterwards
	MPI_Type_free(&full_tile);
	MPI_Type_free(&right_tile);
	MPI_Type_free(&bottom_tile);
	MPI_Type_free(&bottom_right_tile);
}
