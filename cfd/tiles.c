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
	init_tile_shape(nprocs, mesh_width, mesh_height, tile_data);
	init_tile_pos(rank, tile_data);
	init_tile_start_end(tile_data);
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

				// Will fail if the number of procs in the axis is > 1/2 axis length. It is reasonable that this will not take place.
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
	tile_data->pos_x = rank % tile_data->num_x;
	tile_data->pos_y = rank / tile_data->num_x;
}

void init_tile_start_end(struct TileData* tile_data) {
	tile_data->start_x = tile_data->pos_x * tile_data->width;
	tile_data->start_y = tile_data->pos_y * tile_data->height;
	tile_data->end_x = min(tile_data->start_x + tile_data->width, tile_data->mesh_width);
	tile_data->end_y = min(tile_data->start_y + tile_data->height, tile_data->mesh_height);
	tile_data->width = tile_data->end_x - tile_data->start_x;
	tile_data->height = tile_data->end_y - tile_data->start_y;
}

void init_tile_datatypes(struct TileData* tile_data) {
	MPI_Type_contiguous(tile_data->height, MPI_FLOAT, &tile_data->tilecoltype);
    MPI_Type_commit(&tile_data->tilecoltype);
	MPI_Type_vector(tile_data->width, 1, tile_data->mesh_height, MPI_FLOAT, &tile_data->tilerowtype);
    MPI_Type_commit(&tile_data->tilerowtype);
}

void free_tile_data(struct TileData* tile_data) {
	MPI_Type_free(&tile_data->tilecoltype);
    MPI_Type_free(&tile_data->tilerowtype);
}


void halo_sync(int rank, float** array, struct TileData* tile_data, double * sync_time_taken) {
	double start = MPI_Wtime();
	MPI_Request requests[16];
	int requests_pos = 0;
	// No -1 after end_? as the range is non inclusive, so end_? is from the next tile.
//      MPI_Barrier(MPI_COMM_WORLD);
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
	
	// Diagonals
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

	// Diagonals
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
	MPI_Datatype full_tile, right_tile, bottom_tile, bottom_right_tile;
	int right_tile_width = tile_data->std_width- (tile_data->std_width * tile_data->num_x - tile_data->mesh_width);
	int bottom_tile_height = tile_data->std_height-(tile_data->std_height * tile_data->num_y - tile_data->mesh_height);
	MPI_Type_vector(tile_data->std_width, tile_data->std_height, tile_data->mesh_height, MPI_FLOAT, &full_tile);
    MPI_Type_commit(&full_tile);

	MPI_Type_vector(right_tile_width, tile_data->std_height, tile_data->mesh_height, MPI_FLOAT, &right_tile);
    MPI_Type_commit(&right_tile);

	MPI_Type_vector(tile_data->std_width, bottom_tile_height, tile_data->mesh_height, MPI_FLOAT, &bottom_tile);
    MPI_Type_commit(&bottom_tile);

	MPI_Type_vector(right_tile_width, bottom_tile_height, tile_data->mesh_height, MPI_FLOAT, &bottom_right_tile);
    MPI_Type_commit(&bottom_right_tile);
	int nprocs = tile_data->num_x * tile_data->num_y;
	MPI_Request* requests = malloc(sizeof(MPI_Request) * nprocs);

	if (rank == 0) {
		for (int other_rank = 1; other_rank < nprocs; other_rank++) {
			int other_pos_x = other_rank % tile_data->num_x;
			int other_pos_y = other_rank / tile_data->num_x;

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
			int other_start_x = other_pos_x * tile_data->std_width;
			int other_start_y = other_pos_y * tile_data->std_height;
			MPI_Irecv(&array[other_start_x][other_start_y], 1, *dtype, other_rank, 0, MPI_COMM_WORLD, &requests[other_rank]);
		}

		MPI_Status s;
		for (int i = 1; i < nprocs; i++) {
			MPI_Wait(&requests[i], &s);
		}
		printf("Recv ok\n");
		// Async receive
	} else {
		// Async send
		MPI_Datatype* dtype = &full_tile;
		int on_right = tile_data->pos_x == tile_data->num_x - 1;
		int on_bottom = tile_data->pos_y == tile_data->num_y - 1;
		if (on_right && on_bottom) {
			dtype = &bottom_right_tile;
		} else if (on_right) {
			dtype = &right_tile;
		} else if (on_bottom) {
			dtype = &bottom_tile;
		}
		MPI_Send(&array[tile_data->start_x][tile_data->start_y], 1, *dtype, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Type_free(&full_tile);
	MPI_Type_free(&right_tile);
	MPI_Type_free(&bottom_tile);
	MPI_Type_free(&bottom_right_tile);
}

void screw_it_sync_everything(int rank, float** array, struct TileData* tile_data) {
	sync_tile_to_root(rank, array, tile_data);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Datatype full_matrix;
	MPI_Type_contiguous(tile_data->mesh_height * tile_data->mesh_width, MPI_FLOAT, &full_matrix);
    MPI_Type_commit(&full_matrix);

	if (rank == 0) {
		int nprocs = tile_data->num_x * tile_data->num_y;
		MPI_Request* requests = malloc(sizeof(MPI_Request) * nprocs);
		for (int other_rank = 1; other_rank < nprocs; other_rank++) {
			MPI_Isend(&array[0][0], 1, full_matrix, other_rank, 0, MPI_COMM_WORLD, &requests[other_rank]);
		}

		MPI_Status s;
		for (int i = 1; i < nprocs; i++) {
			MPI_Wait(&requests[i], &s);
		}
		printf("send ok\n");
		// Async receive
	} else {
		MPI_Status s;
		MPI_Recv(&array[0][0], 1, full_matrix, 0, 0, MPI_COMM_WORLD, &s);
	}

	MPI_Type_free(&full_matrix);
	MPI_Barrier(MPI_COMM_WORLD);
}

void print_tile(float** array, struct TileData* tile_data)
{
    int i, j;
    for ( i = tile_data->start_x - 3; i < tile_data->end_x + 3; i++) {
        for ( j = tile_data->start_y - 3; j < tile_data->end_y + 3; j++) {
            printf("%10g, ", array[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix(float** array, int cols, int rows)
{
    int i, j;
	printf("start\n");
    for ( i = 0; i < cols; i++) {
        for ( j = 0; j < rows; j++) {
            printf("%5g, ", array[i][j]);
        }
        printf("\n");
    }
    printf("\nEND\n");
}

void test_halo_sync(int rank, int nprocs) {
	int COLS = 31;
	int ROWS = 31;
	double t = 0.0;
	float **matrix = alloc_floatmatrix(COLS, ROWS);
	struct TileData tile_data;
	int output_proc = 17;
	int i, j;
	init_tile_data(rank, nprocs, COLS, ROWS, &tile_data);
	for (i = max(0, tile_data.start_x); i < min(COLS, tile_data.end_x); i++) {
        for ( j = max(0, tile_data.start_y); j < min(ROWS, tile_data.end_y); j++) {
			// printf("%d, %d\n", i, j);
            matrix[i][j] = (i * ROWS) + j;
        }
    }
	if (rank == output_proc) {
		print_matrix(matrix, COLS, ROWS);
	}
	halo_sync(rank, matrix, &tile_data, &t);
	if (rank == output_proc) {
		print_matrix(matrix, COLS, ROWS);
	}

	for (i = max(0, tile_data.start_x); i < min(COLS, tile_data.end_x); i++) {
        for ( j = max(0, tile_data.start_y); j < min(ROWS, tile_data.end_y); j++) {
            matrix[i][j] *= -1;
        }
    }
	if (rank == output_proc) {
		print_matrix(matrix, COLS, ROWS);
	}
	halo_sync(rank, matrix, &tile_data, &t);
	if (rank == output_proc) {
		print_matrix(matrix, COLS, ROWS);
	}

	sync_tile_to_root(rank, matrix, &tile_data);
	if (rank == 0) {
		print_matrix(matrix, COLS, ROWS);
	}

	free_matrix(matrix);
}
