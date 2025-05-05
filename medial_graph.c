#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

int next_crossing(Knot *k, int idx) {
    int i_idx = idx / 4;
    int j_idx = idx - i_idx * 4;
    int next_in_crossing = i_idx * 4 + (j_idx + 1) % 4;
    int segment = k->crossings[next_in_crossing];
    for(int i = 0; i < k->num_crossings * 4; i++){
        if((k->crossings[i] == segment) && (i != next_in_crossing)){
            return i;
        }
    }
    perror("Error: next_crossing can only find segment once.");
    return -1;
}

typedef struct edge {
    int source;
    int target;
    int sign;
    int crossing;
} Edge;

void free_edges(Edge **edges, int num_edges){
    for(int i = 0; i < num_edges; i++) {
        free(edges[i]);
    }
    free(edges);
}

int edge_cmp(const void *a, const void *b){
    Edge *edge1 = (Edge *)a;
    Edge *edge2 = (Edge *)b;
    if(edge1->source < edge2->source){
        return -1;
    }
    if(edge1->source == edge2->source && edge1->target < edge2->target){
        return -1;
    }
    if(edge1->source == edge2->source && edge1->target == edge2->target && edge1->sign < edge2->sign){
        return -1;
    }
    return 1;
}

MedialGraph *getMedialGraph(Knot *knot){
    int *faces = malloc((size_t)knot->num_crossings * 4 * sizeof(int));
    int max_face = 1;
    for(int i = 0; i < knot->num_crossings * 4; i++) {
        faces[i] = 0;
    }
    for(int i = 0; i < knot->num_crossings * 4; i++) {
        if(faces[i] == 0){
            int idx = i;
            do {
                faces[idx] = max_face;
                idx = next_crossing(knot, idx);
                if(idx == -1){
                    free(faces);
                    return NULL;
                }
            } while(idx != i);
            max_face++;
        }
    }

    if(max_face-1 != knot->num_crossings + 2){
        printf("%d, %d\n", max_face, knot->num_crossings + 2);
    }

    Edge **edges = malloc((size_t)knot->num_crossings * 2 * sizeof(Edge *));
    for(int i = 0; i < knot->num_crossings; i++) {
        edges[2 * i] = malloc(sizeof(Edge));
        edges[2 * i + 1] = malloc(sizeof(Edge));
        if(faces[4 * i] < faces[4 * i + 2]){
            edges[2 * i]->source = faces[4 * i];
            edges[2 * i]->target =  faces[4 * i + 2];
        } else {
            edges[2 * i]->source = faces[4 * i + 2];
            edges[2 * i]->target =  faces[4 * i];
        }
        edges[2 * i]->sign = -1;
        edges[2 * i]->crossing = i;
        if(faces[4 * i + 1] < faces[4 * i + 3]){
            edges[2 * i + 1]->source = faces[4 * i + 1];
            edges[2 * i + 1]->target = faces[4 * i + 3];
        } else {
            edges[2 * i + 1]->source = faces[4 * i + 3];
            edges[2 * i + 1]->target = faces[4 * i + 1];
        }
        edges[2 * i + 1]->sign = 1;
        edges[2 * i + 1]->crossing = i;
    }
    //qsort(edges, knot->num_crossings * 2, sizeof(Edge), edge_cmp);
    //for(int i = 1; i < knot->num_crossings * 2; i++) {
    //  if(edges[i-1]->source == edges[i]->source && edges[i-1]->target == edges[i]->target && edges[i-1]->source != edges[i]->source){
    //      remember to de-flype edges[i-1]->crossing and edges[i]->crossing
    //  }
    //}
    // output should be: list of flyping-pairs
    // check that no crossing is in two deflypes
    // if flypings is possible:
    // {
    // call de-flyping method here and loop back only if no flypes were found
    // free_edges(edges, knot->num_crossings * 2);
    // free(faces);
    //}

    MedialGraph *mg = malloc(sizeof(MedialGraph));
    mg->source = malloc((size_t)knot->num_crossings * 2 * sizeof(int));
    mg->target = malloc((size_t)knot->num_crossings * 2 * sizeof(int));
    mg->sign = malloc((size_t)knot->num_crossings * 2 * sizeof(int));
    mg->parents = malloc((size_t)knot->num_parents * sizeof(int));
    for(int i = 0; i < mg->num_parents; i++){
        mg->parents[i] = knot->parents[i];
    }
    for(int i = 0; i < knot->num_crossings * 2; i++) {
        mg->source[i] = edges[i]->source;
        mg->target[i] = edges[i]->target;
        mg->sign[i] = edges[i]->sign;
    }

    mg->num_faces = knot->num_crossings + 2;
    mg->num_edges = knot->num_crossings * 2;
    mg->prime = knot->prime;
    mg->id = -1;
    mg->num_parents = knot->num_parents;

    free_edges(edges, knot->num_crossings * 2);
    free(faces);
    return mg;
}

MedialGraph **getMedialGraphs(Knot **knots, int num_knots) {
    clock_t start = clock();
    int total_crossings = 0;

    MedialGraph **mgs = malloc((size_t)num_knots * sizeof(MedialGraph *));
    for(int i = 0; i < num_knots; i++){
        mgs[i] = getMedialGraph(knots[i]);
        total_crossings += knots[i]->num_crossings;
    }
    printf("Avg crossings: %f\n", (double)total_crossings / num_knots);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    double time_per_knot = time_taken / num_knots;
    printf("Turned %d knots into their medial graphs in %f s (%f s per knot)\n", num_knots, time_taken, time_per_knot);

    return mgs;
}

void print_mg(MedialGraph *mg) {
    printf("Parents: ");
    for(int i = 0; i < mg->num_parents; i++) {
        printf("%d ", mg->parents[i]);
    }
    for(int i = 0; i < mg->num_edges; i++) {
        printf("%d -> (%d) -> %d\n", mg->source[i], mg->sign[i], mg->target[i]);
    }
}

void free_mg(MedialGraph *mg) {
    free(mg->source);
    free(mg->target);
    free(mg->sign);
    free(mg->parents);
    free(mg);
}

void free_mgs(MedialGraph **mgs, int num_graphs){
    for(int i = 0; i < num_graphs; i++) {
        free_mg(mgs[i]);
    }
    free(mgs);
}
