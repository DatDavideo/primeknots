#pragma once
#define CACHED_TABLE 1
#define MAGIC 0xFFFFFFFF

typedef struct Knot {
    int *crossings;
    int num_crossings;
    int *parents;
    int num_parents;
    int prime;
    #if CACHED_TABLE 
    unsigned int *lookup_table;
    #endif
} Knot;

typedef struct MedialGraph{
    int *source;
    int *target;
    int *sign;
    int num_faces;
    int num_edges;
    int *parents;
    int num_parents;
    int prime;
    int id;
} MedialGraph;

typedef struct Face{
    int *segments;
    int *crossings;
    int numSegments;
} Face;

Knot **create_dataset(Knot **primes, int *num_primes, int num_alts, int padding);
MedialGraph *getMedialGraph(Knot *knot);
MedialGraph **getMedialGraphs(Knot **knots, int num_knots);

void print_mg(MedialGraph *mg);
void print_k(Knot *k);

void write_to_file(MedialGraph **mgs, int num_graphs, const char *filename);
Knot **read_prime_knots(const char *filename, int *num_lines);

Knot *copy_knot(Knot *k);

void free_mg(MedialGraph *mgs);
void free_mgs(MedialGraph **mgs, int num_graphs);
void free_knot(Knot *knot);
void free_knots(Knot **knots, int num_knots);
void free_face(Face *face);

#if CACHED_TABLE
void create_lookup_table(Knot *k);
#endif

int compare(const void *a, const void *b);

Face *get_face(Knot *k, int segment, int direction, int max_segments);

int same_direction(Knot *k, int segment);
int different_segments(Face *face);

void generate_ids(MedialGraph **graphs, int num_graphs);
