#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <mpi.h>
#include <immintrin.h>
#include "alloc.h"
#include "boundary.h"
#include "datadef.h"
#include "init.h"
#include "simulation-par.h"
#include "tiles.h"

void write_bin(float **u, float **v, float **p, char **flag,
     int imax, int jmax, float xlength, float ylength, char *file);

int read_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char *file);

static void print_usage(void);
static void print_version(void);
static void print_help(void);

static char *progname;

int proc = 0;                       /* Rank of the current process */
int nprocs = 0;                /* Number of processes in communicator */

int *ileft, *iright;           /* Array bounds for each processor */

#define PACKAGE "karman"
#define VERSION "1.0"

/* Command line options */
static struct option long_opts[] = {
    { "del-t",   1, NULL, 'd' },
    { "help",    0, NULL, 'h' },
    { "imax",    1, NULL, 'x' },
    { "infile",  1, NULL, 'i' },
    { "jmax",    1, NULL, 'y' },
    { "outfile", 1, NULL, 'o' },
    { "t-end",   1, NULL, 't' },
    { "verbose", 1, NULL, 'v' },
    { "version", 1, NULL, 'V' },
    { 0,         0, 0,    0   } 
};
#define GETOPTS "d:hi:o:t:v:Vx:y:"

int main(int argc, char *argv[])
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    int verbose = 1;          /* Verbosity level */
    float xlength = 22.0;     /* Width of simulated domain */
    float ylength = 4.1;      /* Height of simulated domain */
    int imax = 660;           /* Number of cells horizontally */
    int jmax = 120;           /* Number of cells vertically */

    char *infile;             /* Input raw initial conditions */
    char *outfile;            /* Output raw simulation results */

    float t_end = 2.1;        /* Simulation runtime */
    float del_t = 0.003;      /* Duration of each timestep */
    float tau = 0.5;          /* Safety factor for timestep control */

    int itermax = 100;        /* Maximum number of iterations in SOR */
    float eps = 0.001;        /* Stopping error threshold for SOR */
    float omega = 1.7;        /* Relaxation parameter for SOR */
    float gamma = 0.9;        /* Upwind differencing factor in PDE
                                 discretisation */

    float Re = 150.0;         /* Reynolds number */
    float ui = 1.0;           /* Initial X velocity */
    float vi = 0.0;           /* Initial Y velocity */

    float t, delx, dely;
    int  i, j, itersor = 0, ifluid = 0, ibound = 0;
    float res;
    float **u, **v, **p, **rhs, **f, **g;
    char  **flag;
    int init_case, iters = 0;
    int show_help = 0, show_usage = 0, show_version = 0;

    progname = argv[0];
    infile = strdup("karman.bin");
    outfile = strdup("karman.bin");

    int optc;
    while ((optc = getopt_long(argc, argv, GETOPTS, long_opts, NULL)) != -1) {
        switch (optc) {
            case 'h':
                show_help = 1;
                break;
            case 'V':
                show_version = 1;
                break;
            case 'v':
                verbose = atoi(optarg);
                break;
            case 'x':
                imax = atoi(optarg);
                break;
            case 'y':
                jmax = atoi(optarg);
                break;
            case 'i':
                free(infile);
                infile = strdup(optarg);
                break;
            case 'o':
                free(outfile);
                outfile = strdup(optarg);
                break;
            case 'd':
                del_t = atof(optarg);
                break;
            case 't':
                t_end = atof(optarg);
                break;
            default:
                show_usage = 1;
        }
    }
    if (show_usage || optind < argc) {
        print_usage();
        return 1;
    }
    
    if (show_version) {
        print_version();
        if (!show_help) {
            return 0;
        }
    }
    
    if (show_help) {
        print_help();
        return 0;
    }

    double program_start = MPI_Wtime();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    // test_halo_sync(proc,  nprocs);

    delx = xlength/imax;
    dely = ylength/jmax;

    /* Allocate arrays */
    u    = alloc_floatmatrix(imax+2, jmax+2);
    v    = alloc_floatmatrix(imax+2, jmax+2);
    f    = alloc_floatmatrix(imax+2, jmax+2);
    g    = alloc_floatmatrix(imax+2, jmax+2);
    p    = alloc_floatmatrix(imax+2, jmax+2);
    rhs  = alloc_floatmatrix(imax+2, jmax+2); 
    flag = alloc_charmatrix(imax+2, jmax+2);      

    // Initialise the tile data
    struct TileData tile_data;
    init_tile_data(proc, nprocs, imax + 2, jmax + 2, &tile_data);
    // Output the info for debug purposes
    if (proc == 0) {
        printf("I am process %d. The mesh has been split into a shape of %dx%d. Tiles sizes are %dx%d\n. My pos is %dx%d. Array access is [%d:%d, %d:%d]\n"
        , proc, 
        tile_data.num_x, 
        tile_data.num_y, 
        tile_data.width, 
        tile_data.height, 
        tile_data.pos_x, 
        tile_data.pos_y,
        tile_data.start_x,
        tile_data.end_x,
        tile_data.start_y,
        tile_data.end_y);
    }

    if (!u || !v || !f || !g || !p || !rhs || !flag) {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
        return 1;
    }

    /* Read in initial values from a file if it exists */
    init_case = read_bin(u, v, p, flag, imax, jmax, xlength, ylength, infile);
        
    if (init_case > 0) {
        /* Error while reading file */
        return 1;
    }

    if (init_case < 0) {
        /* Set initial values if file doesn't exist */
        #pragma omp parallel for private(i, j) firstprivate(ui, vi, u, v, p, imax, jmax) default(none)
        for (i=0;i<=imax+1;i++) {
            for (j=0;j<=jmax+1;j++) {
                u[i][j] = ui;
                v[i][j] = vi;
                p[i][j] = 0.0;
            }
        }
        init_flag(flag, imax, jmax, delx, dely, &ibound);
        apply_tile_boundary_conditions(u, v, flag, imax, jmax, ui, vi, &tile_data);
        double sync_time_taken = 0.0;
        halo_sync(proc, u, &tile_data, &sync_time_taken);
        halo_sync(proc, v, &tile_data, &sync_time_taken);
        halo_sync(proc, p, &tile_data, &sync_time_taken);
    }

    // Variables to keep track of time taken for each part
    double start, timestep_time_taken, compute_velocity_time_taken, rhs_time_taken, possion_time_taken, update_velocity_time_taken, boundary_time_taken;
    double sync_time_taken, possion_p_loop_time_taken, possion_res_loop_time_taken = 0.0;
    /* Main loop */
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        // Reset the cumulative time variables for each new loop iteration
        sync_time_taken = 0.0;
        possion_p_loop_time_taken = 0.0;
        possion_res_loop_time_taken = 0.0;

        start = MPI_Wtime();
        set_timestep_interval(&del_t, imax, jmax, delx, dely, u, v, Re, tau, &tile_data, &sync_time_taken);
        timestep_time_taken = MPI_Wtime() - start;

        ifluid = (imax * jmax) - ibound;

        start = MPI_Wtime();
        compute_tentative_velocity(u, v, f, g, flag, imax, jmax,
            del_t, delx, dely, gamma, Re, &tile_data, &sync_time_taken);
        compute_velocity_time_taken = MPI_Wtime() - start;

        start = MPI_Wtime();
        compute_rhs(f, g, rhs, flag, imax, jmax, del_t, delx, dely, &tile_data);
        rhs_time_taken = MPI_Wtime() - start;

        start = MPI_Wtime();
        if (ifluid > 0) {
            itersor = poisson(p, rhs, flag, imax, jmax, delx, dely,
                    eps, itermax, omega, &res, ifluid, &tile_data, &sync_time_taken, &possion_p_loop_time_taken, &possion_res_loop_time_taken);
                    //screw_it_sync_everything(proc, u, &tile_data, &sync_time_taken);
        } else {
            itersor = 0;
        }
        possion_time_taken = MPI_Wtime() - start;

        if (proc == 0 && verbose > 1) {
            printf("%d t:%g, del_t:%g, SOR iters:%3d, res:%e, bcells:%d\n",
                iters, t+del_t, del_t, itersor, res, ibound);
        }
        start = MPI_Wtime();
        update_velocity(u, v, f, g, p, flag, imax, jmax, del_t, delx, dely, &tile_data, &sync_time_taken);
        update_velocity_time_taken = MPI_Wtime() - start;

        start = MPI_Wtime();
        apply_tile_boundary_conditions(u, v, flag, imax, jmax, ui, vi, &tile_data);
        halo_sync(proc, u, &tile_data, &sync_time_taken);
        halo_sync(proc, v, &tile_data, &sync_time_taken);
        boundary_time_taken = MPI_Wtime() - start;

        // Calculate the average time spent syncing
        double avg_sync_time_taken = 1.0;
        MPI_Reduce(&sync_time_taken, &avg_sync_time_taken, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        avg_sync_time_taken = avg_sync_time_taken / (double) nprocs;
        if (proc == 0) {
            // Output timing data
            printf("\n --- Timestep %f of %f ---\n", t, t_end);
            printf("timestep_time_taken: %f\n", timestep_time_taken);
            printf("compute_velocity_time_taken: %f\n", compute_velocity_time_taken);
            printf("rhs_time_taken: %f\n", rhs_time_taken);
            printf("possion_time_taken: %f\n", possion_time_taken);
            printf("update_velocity_time_taken: %f\n", update_velocity_time_taken);
            printf("boundary_time_taken: %f\n", boundary_time_taken);

            printf("sync_time_taken: %f\n", avg_sync_time_taken);
            printf("possion_p_loop_time_taken: %f\n", possion_p_loop_time_taken);
            printf("possion_res_loop_time_taken: %f\n", possion_res_loop_time_taken);
        }
    } /* End of main loop */
    
    // Copy all the tiles to the root node (0) to save the output
    sync_tile_to_root(proc, u, &tile_data);
    sync_tile_to_root(proc, v, &tile_data);
    sync_tile_to_root(proc, f, &tile_data);
    sync_tile_to_root(proc, g, &tile_data);
    sync_tile_to_root(proc, p, &tile_data);
  
    if (outfile != NULL && strcmp(outfile, "") != 0 && proc == 0) {
        write_bin(u, v, p, flag, imax, jmax, xlength, ylength, outfile);
    }

    free_matrix(u);
    free_matrix(v);
    free_matrix(f);
    free_matrix(g);
    free_matrix(p);
    free_matrix(rhs);
    free_matrix(flag);
    free_tile_data(&tile_data);

    printf("Program total time: %f", MPI_Wtime() - program_start);
    MPI_Finalize();
    return 0;
}

/* Save the simulation state to a file */
void write_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char* file)
{
    int i;
    FILE *fp;

    fp = fopen(file, "wb"); 

    if (fp == NULL) {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
            strerror(errno));
        return;
    }

    fwrite(&imax, sizeof(int), 1, fp);
    fwrite(&jmax, sizeof(int), 1, fp);
    fwrite(&xlength, sizeof(float), 1, fp);
    fwrite(&ylength, sizeof(float), 1, fp);

    for (i=0;i<imax+2;i++) {
        fwrite(u[i], sizeof(float), jmax+2, fp);
        fwrite(v[i], sizeof(float), jmax+2, fp);
        fwrite(p[i], sizeof(float), jmax+2, fp);
        fwrite(flag[i], sizeof(char), jmax+2, fp);
    }
    printf("%f %f %f\n", v[170][20], v[169][20], v[171][20]);
    fclose(fp);
}

/* Read the simulation state from a file */
int read_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char* file)
{
    int i,j;
    FILE *fp;

    if (file == NULL) return -1;

    if ((fp = fopen(file, "rb")) == NULL) {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
            strerror(errno));
        fprintf(stderr, "Generating default state instead.\n");
        return -1;
    }

    fread(&i, sizeof(int), 1, fp);
    fread(&j, sizeof(int), 1, fp);
    float xl, yl;
    fread(&xl, sizeof(float), 1, fp);
    fread(&yl, sizeof(float), 1, fp);

    if (i!=imax || j!=jmax) {
        fprintf(stderr, "Warning: imax/jmax have wrong values in %s\n", file);
        fprintf(stderr, "%s's imax = %d, jmax = %d\n", file, i, j);
        fprintf(stderr, "Program's imax = %d, jmax = %d\n", imax, jmax);
        return 1;
    }
    if (xl!=xlength || yl!=ylength) {
        fprintf(stderr, "Warning: xlength/ylength have wrong values in %s\n", file);
        fprintf(stderr, "%s's xlength = %g,  ylength = %g\n", file, xl, yl);
        fprintf(stderr, "Program's xlength = %g, ylength = %g\n", xlength,
            ylength);
        return 1;
    }

    for (i=0; i<imax+2; i++) {
        fread(u[i], sizeof(float), jmax+2, fp);
        fread(v[i], sizeof(float), jmax+2, fp);
        fread(p[i], sizeof(float), jmax+2, fp);
        fread(flag[i], sizeof(char), jmax+2, fp);
    }
    fclose(fp);
    return 0;
}

static void print_usage(void)
{
    fprintf(stderr, "Try '%s --help' for more information.\n", progname);
}

static void print_version(void)
{
    fprintf(stderr, "%s %s\n", PACKAGE, VERSION);
}

static void print_help(void)
{
    fprintf(stderr, "%s. A simple computational fluid dynamics tutorial.\n\n",
        PACKAGE);
    fprintf(stderr, "Usage: %s [OPTIONS]...\n\n", progname);
    fprintf(stderr, "  -h, --help            Print a summary of the options\n");
    fprintf(stderr, "  -V, --version         Print the version number\n");
    fprintf(stderr, "  -v, --verbose=LEVEL   Set the verbosity level. 0 is silent\n");
    fprintf(stderr, "  -x, --imax=IMAX       Set the number of interior cells in the X direction\n");
    fprintf(stderr, "  -y, --jmax=JMAX       Set the number of interior cells in the Y direction\n");
    fprintf(stderr, "  -t, --t-end=TEND      Set the simulation end time\n");
    fprintf(stderr, "  -d, --del-t=DELT      Set the simulation timestep size\n");
    fprintf(stderr, "  -i, --infile=FILE     Read the initial simulation state from this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
    fprintf(stderr, "  -o, --outfile=FILE    Write the final simulation state to this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
}
