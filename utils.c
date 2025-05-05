#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

int compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

int compare_graphs(const void *a, const void *b){
    const MedialGraph *mg1 = *(const MedialGraph **)a;
    const MedialGraph *mg2 = *(const MedialGraph **)b;

    if(mg1->num_parents > mg2->num_parents){
        return 1;
    } else if(mg1->num_parents < mg2->num_parents){
        return -1;
    } else {
        for(int i = 0; i < mg1->num_parents; i++){
            if(mg1->parents[i] > mg2->parents[i]){
                return 1;
            } else if(mg1->parents[i] < mg2->parents[i]){
                return -1;
            } 
        }
        return 0;
    }
}

int compare_graphs2(MedialGraph *mg1, MedialGraph *mg2){
    if(mg1->num_parents > mg2->num_parents){
        return 1;
    } else if(mg1->num_parents < mg2->num_parents){
        return -1;
    } else {
        for(int i = 0; i < mg1->num_parents; i++){
            if(mg1->parents[i] > mg2->parents[i]){
                return 1;
            } else if(mg1->parents[i] < mg2->parents[i]){
                return -1;
            } 
        }
        return 0;
    }
}

void free_knot(Knot *knot) {
    free(knot->crossings);
    free(knot->parents);
    #if CACHED_TABLE
    if(knot->lookup_table != NULL){
        free(knot->lookup_table);
    }
    #endif
    free(knot);
}

void free_knots(Knot **knots, int num_knots){
    for(int i = 0; i < num_knots; i++) {
        free_knot(knots[i]);
    }
    free(knots);
}

void free_face(Face *face){
    free(face->segments);
    free(face->crossings);
    free(face);
}

void print_k(Knot *k) {
    printf("Parents: ");
    for(int i = 0; i < k->num_parents; i++) {
        printf("%d ", k->parents[i]);
    }
    printf("\n");
    for(int i = 0; i < k->num_crossings; i++) {
        printf("    Crossing %d: ", i);
        for(int j = 0; j < 4; j++) {
            printf("%d ", k->crossings[i * 4 + j]);
        }
        printf("\n");
    }
}

Knot *copy_knot(Knot *k){
    Knot *copy = malloc(sizeof(Knot));
    copy->num_parents = k->num_parents;
    copy->parents = malloc((size_t)copy->num_parents * sizeof(int));
    for(int i = 0; i < k->num_parents; i++) {
        copy->parents[i] = k->parents[i];
    }
    copy->prime = k->prime;
    copy->num_crossings = k->num_crossings;
    copy->crossings = malloc((size_t)copy->num_crossings * 4 * sizeof(int));
    for(int i = 0; i < k->num_crossings * 4; i++) {
        copy->crossings[i] = k->crossings[i];
    }
    return copy;
}

#if CACHED_TABLE
void create_lookup_table(Knot *k){
    if(k->lookup_table != NULL) {
        free(k->lookup_table);
    }
    k->lookup_table = malloc((size_t)k->num_crossings * 4 * sizeof(int));
    for(int i = 0; i < k->num_crossings * 4; i++){
        k->lookup_table[i] = MAGIC;
    }
    for(unsigned int i = 0; i < (unsigned int)k->num_crossings * 4; i++){
        if(k->lookup_table[2 * (k->crossings[i] - 1)] == MAGIC){
            k->lookup_table[2 * (k->crossings[i] - 1)] = i;
        } else {
            k->lookup_table[2 * (k->crossings[i] - 1) + 1] = i;
        }
    }
}
#endif

#if CACHED_TABLE
Face *get_face(Knot *k, int segment, int direction, int max_segments){
    Face *face = malloc(sizeof(Face));
    face->segments = malloc((size_t)k->num_crossings * 2 * sizeof(int));
    face->crossings = malloc((size_t)k->num_crossings * 2 * sizeof(int));

    face->numSegments = 0;
    int current_segment = segment;
    unsigned int prev_crossing = MAGIC;
    do{
        face->segments[face->numSegments] = current_segment;
        if(k->lookup_table[2 * (current_segment - 1)] == prev_crossing){
            face->crossings[face->numSegments] = (int)k->lookup_table[2 * (current_segment - 1) + 1];
        } else {
            face->crossings[face->numSegments] = (int)k->lookup_table[2 * (current_segment - 1)];
        }
        int i_idx = face->crossings[face->numSegments] / 4;
        int j_idx = face->crossings[face->numSegments] % 4;
        face->numSegments++;
        if(direction){
            prev_crossing = (unsigned int)(i_idx * 4 + ((j_idx + 1) % 4));
            current_segment = k->crossings[prev_crossing];
        } else {
            prev_crossing = (unsigned int)(i_idx * 4 + ((j_idx + 3) % 4));
            current_segment = k->crossings[prev_crossing];
        }
        if(max_segments != -1 && face->numSegments > max_segments){
            free_face(face);
            return NULL;
        }
    } while(segment != current_segment);
    return face;
}
#else
Face *get_face(Knot *k, int segment, int direction, int max_segments){
    Face *face = malloc(sizeof(Face));
    face->segments = malloc((size_t)k->num_crossings * 2 * sizeof(int));
    face->crossings = malloc((size_t)k->num_crossings * 2 * sizeof(int));
    face->numSegments = 0;
    int current_segment = segment;
    unsigned int prev_crossing = MAGIC;
    do{
        for(int i = 0; i < k->num_crossings * 4; i++){
            if(prev_crossing != (unsigned int)i && k->crossings[i] == current_segment){
                face->segments[face->numSegments] = current_segment;
                face->crossings[face->numSegments] = i;
                face->numSegments++;
                int i_idx = i / 4;
                int j_idx = i % 4;
                if(direction){
                    prev_crossing = (unsigned int)(i_idx * 4 + ((j_idx + 1) % 4));
                    current_segment = k->crossings[prev_crossing];
                } else {
                    prev_crossing = (unsigned int)(i_idx * 4 + ((j_idx + 3) % 4));
                    current_segment = k->crossings[prev_crossing];
                }
                break;
            }
        }
        if(max_segments != -1 && face->numSegments > max_segments){
            free_face(face);
            return NULL;
        }
    } while(segment != current_segment);
    return face;
}
#endif


int same_direction(Knot *k, int segment){
    int dir = 0;
    for(int i = 0; i < k->num_crossings * 4; i++) {
        if(k->crossings[i] == segment){
            if(i % 4 == 0 || i % 4 == 2) {
                dir += 1;
            }
        }
    }
    return dir != 1;
}

int different_segments(Face *face){
    for(int i = 0; i < face->numSegments; i++){
        for(int j = i + 1; j < face->numSegments; j++){
            if((face->crossings[i] / 4) == (face->crossings[j] / 4)|| face->segments[i] == face->segments[j]){
                return 0;
            }
        }
    }
    return 1;
}

void generate_ids(MedialGraph **graphs, int num_graphs){
    clock_t start = clock();

    for(int i = 0; i < num_graphs; i++){
        qsort(graphs[i]->parents, (size_t)graphs[i]->num_parents, sizeof(int), compare);
    }
    qsort(graphs, (size_t)num_graphs, sizeof(MedialGraph *), compare_graphs);
    // printf("Sorted\n");
    int id = 1;
    graphs[0]->id = id;
    for(int i = 1; i < num_graphs; i++){
        if(compare_graphs2(graphs[i], graphs[i - 1]) != 0){
            id++;
        }
        graphs[i]->id = id;
        // printf("ID: %d\n", id);
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    double time_avg = time_taken / num_graphs;
    printf("Created %d ids in %fs (on average %fs per id created).\n", num_graphs, time_taken, time_avg);
}
