#define t_start 0.0001
#define t_byte 0.0001
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
	tile_data->start_x = tile_data->pos_x * tile_data->width;	// TODO problably need to update width for edge mesh parts
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


void halo_sync(int rank, float** array, struct TileData* tile_data) {
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
	// printf("Sync for %d took %f seconds\n", rank, MPI_Wtime() - start);
	// MPI_Barrier(MPI_COMM_WORLD);
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
			// printf("Receiving %d with pos %dx%d. Location vals r%d, b%d, rr%d, bb%d. Start is %dx%d.\n", rank, other_pos_x, other_pos_y, on_right, on_bottom, right_tile_width, bottom_tile_height, other_start_x, other_start_y);
			// printf("%ld %ld", &array[0][0], &array[0][1]);
			MPI_Irecv(&array[other_start_x][other_start_y], 1, *dtype, other_rank, 0, MPI_COMM_WORLD, &requests[other_rank]);

			// MPI_Status s;
			// MPI_Recv(&array[other_start_x][other_start_y], 1, *dtype, other_rank, 0, MPI_COMM_WORLD, &s);
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

void print_tile(float** array, struct TileData* tile_data)
{
    int i, j;
    // printf("%s\n", title);
    for ( i = tile_data->start_x; i < tile_data->end_x; i++) {
        for ( j = tile_data->start_y; j < tile_data->end_y; j++) {
            printf("%10g ", array[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix(float** array, int cols, int rows)
{
    int i, j;
	printf("start\n");
    // printf("%s\n", title);
    for ( i = 0; i < cols; i++) {
        for ( j = 0; j < rows; j++) {
            printf("%5g ", array[i][j]);
        }
        printf("\n");
    }
    printf("\nEND\n");
}

void test_halo_sync(int rank, int nprocs) {
	int COLS = 12;
	int ROWS = 24;
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
	halo_sync(rank, matrix, &tile_data);
	if (rank == output_proc) {
		print_matrix(matrix, COLS, ROWS);
	}

	for (i = max(0, tile_data.start_x); i < min(COLS, tile_data.end_x); i++) {
        for ( j = max(0, tile_data.start_y); j < min(ROWS, tile_data.end_y); j++) {
			// printf("%d, %d\n", i, j);
            matrix[i][j] *= -1;
        }
    }
	if (rank == output_proc) {
		print_matrix(matrix, COLS, ROWS);
	}
	halo_sync(rank, matrix, &tile_data);
	if (rank == output_proc) {
		print_matrix(matrix, COLS, ROWS);
	}

	free_matrix(matrix);
}

// int min(int a, int b) {
// 	return (a > b) ? b : a;
// }