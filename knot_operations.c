#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

Knot *knot_sum(Knot *k1, Knot *k2){
    int numSegments1 = k1->num_crossings * 2;
    int numSegments2 = k2->num_crossings * 2;

    int k1_segment = (rand() % numSegments1) + 1;
    int k2_segment = (rand() % numSegments2) + 1;

    Knot *sum = malloc(sizeof(Knot));
    sum->num_parents = k1->num_parents + k2->num_parents;
    sum->parents = malloc((size_t)sum->num_parents * sizeof(int));
    sum->prime = 0;
    sum->num_crossings = k1->num_crossings + k2->num_crossings;
    sum->crossings = malloc((size_t)sum->num_crossings * 4 * sizeof(int));
    for(int i = 0; i < k1->num_parents; i++) {
        sum->parents[i] = k1->parents[i];
    }
    for(int i = 0; i < k2->num_parents; i++) {
        sum->parents[i + k1->num_parents] = k2->parents[i];
    }

    int crossing_set = 0;
    for (int i = 0; i < k1->num_crossings * 4; i++){
        if(!crossing_set && k1->crossings[i] == k1_segment){
            sum->crossings[i] = k2_segment + numSegments1;
            crossing_set = 1;
        } else {
            sum->crossings[i] = k1->crossings[i];
        }
    }
    crossing_set = 0;
    for(int i = 0; i < k2->num_crossings * 4; i++){
        if(!crossing_set && k2->crossings[i] == k2_segment){
            sum->crossings[i + k1->num_crossings * 4] = k1_segment;
            crossing_set = 1;
        } else {
            sum->crossings[i + k1->num_crossings * 4] = k2->crossings[i] + numSegments1;
        }
    }
    return sum;
}

Knot *reidemeister_1(Knot *k){
    int numSegments = k->num_crossings * 2;
    int segment = (rand() % numSegments) + 1;
    k->num_crossings += 1;
    k->crossings = realloc(k->crossings, (size_t)k->num_crossings * 4 * sizeof(int));
    if(k->crossings == NULL) {
        perror("Error reallocating knot crossings in rm1.");
        return NULL;
    }
    int first_crossing_set = 0;
    for(int i = 0; i < k->num_crossings * 4; i++){
        if(!first_crossing_set && k->crossings[i] == segment){
            k->crossings[i] = numSegments + 1;
            first_crossing_set = 1;
        }
    }
    switch(rand() % 2){
        case 0:
            k->crossings[k->num_crossings * 4 - 4] = segment;
            k->crossings[k->num_crossings * 4 - 3] = numSegments + 2;
            k->crossings[k->num_crossings * 4 - 2] = numSegments + 2;
            k->crossings[k->num_crossings * 4 - 1] = numSegments + 1;
            break;
        case 1:
            k->crossings[k->num_crossings * 4 - 4] = numSegments + 1;
            k->crossings[k->num_crossings * 4 - 3] = numSegments + 2;
            k->crossings[k->num_crossings * 4 - 2] = numSegments + 2;
            k->crossings[k->num_crossings * 4 - 1] = segment;
            break;
    }
    return k;
}

Knot *reidemeister_2(Knot *k){
    int numSegments = k->num_crossings * 2;
    int segment = (rand() % numSegments) + 1;

    #if CACHED_TABLE
    create_lookup_table(k);
    #endif
    int dir = rand() % 2;
    Face *face = get_face(k, segment, dir, -1);

    if(face->numSegments == 1){
        free_face(face);
        return k;
    }

    k->num_crossings = k->num_crossings + 2;
    k->crossings = realloc(k->crossings, (size_t)k->num_crossings * 4 * sizeof(int));
    if(k->crossings == NULL) {
        perror("Error reallocating knot crossings in rm2.");
        free_face(face);
        return NULL;
    }

    int segment_idx = (rand() % (face->numSegments - 1)) + 1;
    int rm_segment = face->segments[segment_idx];
    int crossing1 = face->crossings[0];
    int crossing2 = face->crossings[segment_idx];
    free_face(face);

    for(int i = 0; i < (k->num_crossings - 2) * 4; i++){
        if(k->crossings[i] == segment) {
            if(i == crossing1) {
                k->crossings[i] = numSegments + 1;
            } else {
                k->crossings[i] = numSegments + 2;
            }
        } else if(k->crossings[i] == rm_segment){
            if(i == crossing2) {
                k->crossings[i] = numSegments + 3; 
            } else {
                k->crossings[i] = numSegments + 4;
            }
        }
    }
    if(rand() % 2){
        if(dir == 0){
            k->crossings[k->num_crossings * 4 - 8] = numSegments + 1;
            k->crossings[k->num_crossings * 4 - 7] = numSegments + 4;
            k->crossings[k->num_crossings * 4 - 6] = segment;
            k->crossings[k->num_crossings * 4 - 5] = rm_segment;

            k->crossings[k->num_crossings * 4 - 4] = numSegments + 2;
            k->crossings[k->num_crossings * 4 - 3] = rm_segment;
            k->crossings[k->num_crossings * 4 - 2] = segment;
            k->crossings[k->num_crossings * 4 - 1] = numSegments + 3;
        } else {
            k->crossings[k->num_crossings * 4 - 8] = numSegments + 1;
            k->crossings[k->num_crossings * 4 - 7] = rm_segment;
            k->crossings[k->num_crossings * 4 - 6] = segment;
            k->crossings[k->num_crossings * 4 - 5] = numSegments + 4;

            k->crossings[k->num_crossings * 4 - 4] = numSegments + 2;
            k->crossings[k->num_crossings * 4 - 3] = numSegments + 3;
            k->crossings[k->num_crossings * 4 - 2] = segment;
            k->crossings[k->num_crossings * 4 - 1] = rm_segment;
        }
    } else {
        if(dir == 0){
            k->crossings[k->num_crossings * 4 - 8] = numSegments + 4;
            k->crossings[k->num_crossings * 4 - 7] = segment;
            k->crossings[k->num_crossings * 4 - 6] = rm_segment;
            k->crossings[k->num_crossings * 4 - 5] = numSegments + 1;

            k->crossings[k->num_crossings * 4 - 4] = numSegments + 3;
            k->crossings[k->num_crossings * 4 - 3] = numSegments + 2;
            k->crossings[k->num_crossings * 4 - 2] = rm_segment;
            k->crossings[k->num_crossings * 4 - 1] = segment;
        } else {
            k->crossings[k->num_crossings * 4 - 8] = numSegments + 4;
            k->crossings[k->num_crossings * 4 - 7] = numSegments + 1;
            k->crossings[k->num_crossings * 4 - 6] = rm_segment;
            k->crossings[k->num_crossings * 4 - 5] = segment;

            k->crossings[k->num_crossings * 4 - 4] = numSegments + 3;
            k->crossings[k->num_crossings * 4 - 3] = segment;
            k->crossings[k->num_crossings * 4 - 2] = rm_segment;
            k->crossings[k->num_crossings * 4 - 1] = numSegments + 2;
        }
    }
    return k;
}

Knot *reidemeister_3(Knot *k){
    int dir = rand() % 2;
    int numSegments = k->num_crossings * 2;
    int current_segment = (rand() % numSegments) + 1;

    Face *rm3face;
    int rm3able = 0;

    #if CACHED_TABLE
    create_lookup_table(k);
    #endif

    for(int i = 1; i <= numSegments; i++){
        if(same_direction(k, current_segment)){
            rm3face = get_face(k, i, dir, 3);
            if(rm3face != NULL) {
                if(rm3face->numSegments != 3 || !different_segments(rm3face)){
                    free_face(rm3face);
                } else {
                    rm3able = 1;
                    break;
                }
            }
        }
        current_segment = (current_segment + 1) % numSegments + 1;
    }
    if(!rm3able){
        return k;
    }

    int crossing1 = rm3face->crossings[0];
    int crossingMiddle = rm3face->crossings[1];
    int crossing2 = rm3face->crossings[2];
    // int z = rm3faces[rm3id]->segments[0];
    int x = rm3face->segments[1];
    int y = rm3face->segments[2];

    free_face(rm3face);

    int cr1_i = crossing1 / 4;
    int cr1_j = crossing1 % 4;
    // int a = k->crossings[cr1_i * 4 + (cr1_j + 2) % 4];
    int b = k->crossings[cr1_i * 4 + (cr1_j + 1 + 2 * dir) % 4];

    int cr2_i = crossing2 / 4;
    int cr2_j = crossing2 % 4;
    int c = k->crossings[cr2_i * 4 + (cr2_j + 2) % 4];
    // int d = k->crossings[cr2_i * 4 + (cr2_j + 1 + 2 * dir) % 4];

    int crm_i = crossingMiddle / 4;
    int crm_j = crossingMiddle % 4;
    int e = k->crossings[crm_i * 4 + (crm_j + 2) % 4];
    int f = k->crossings[crm_i * 4 + (crm_j + 1 + 2 * dir) % 4];

    k->crossings[crossingMiddle] = b;
    k->crossings[crm_i * 4 + (crm_j + 4 - (1 + 2 * dir)) % 4] = c;
    k->crossings[crossing2] = e;
    k->crossings[cr1_i * 4 + (cr1_j + 4 - (1 + 2 * dir)) % 4] = f;
    k->crossings[crm_i * 4 + (crm_j + 2) % 4] = x;
    k->crossings[cr2_i * 4 + (cr2_j + 2) % 4] = x;
    k->crossings[crm_i * 4 + (crm_j + 1 + 2 * dir) % 4] = y;
    k->crossings[cr1_i * 4 + (cr1_j + 1 + 2 * dir) % 4] = y;
    return k;
}

Knot *increase_crossings(Knot *base, int num_crossings, int *num_moves){
    int base_crossings = base->num_crossings;
    while(base_crossings < num_crossings) {
        if(num_crossings - base_crossings >= 2 && rand() % 3 < 2) {
            base = reidemeister_2(base);
            *num_moves += 1;
        }
        if(base_crossings == base->num_crossings) {
            base = reidemeister_1(base);
            *num_moves += 1;
        }
        if(rand() % 3 == 0) {
            base = reidemeister_3(base);
            *num_moves += 1;
        }
        base_crossings = base->num_crossings;
    }
    return base;
}

Knot **create_alters(Knot *base, int num_alts, int num_crossings, int *num_moves) {
    int base_crossings = base->num_crossings;
    int diff = num_crossings - base_crossings;
    if(diff < num_alts) {
        printf("%d, %d\n", base->num_crossings, num_crossings);
        Knot **alts = malloc(sizeof(Knot *));
        alts[0] = base;
        return alts;
    }
    int interval = diff / num_alts;
    Knot **alts = malloc((size_t)num_alts * sizeof(Knot *));
    for(int i = 0; i < num_alts - 1; i++) {
        if(i > 0){
            alts[i] = copy_knot(alts[i - 1]);
        } else {
            alts[i] = copy_knot(base);
        }
        alts[i] = increase_crossings(alts[i], base_crossings + (i + 1) * interval, num_moves);
    }
    alts[num_alts - 1] = copy_knot(base);
    alts[num_alts - 1] = increase_crossings(alts[num_alts - 1], num_crossings, num_moves);
    free_knot(base);

    return alts;
}

Knot **create_dataset(Knot **primes, int *num_primes, int num_alts, int padding){
    clock_t start = clock();

    Knot **dataset = realloc(primes, (size_t)(*num_primes) * 2 * (size_t)num_alts * sizeof(Knot *));
    if(dataset == NULL){
        perror("Error reallocating memory for the dataset.");
        return NULL;
    }

    int max_crossings = 0;
    int total_additions = 0;
    for(int i = 0; i < (*num_primes); i++) {
        int additions = rand() % 5;
        total_additions += additions + 1;
        Knot *sum1 = knot_sum(dataset[i], dataset[rand() % (*num_primes)]);
        for(int j = 0; j < additions; j++){
            Knot *sum2 = knot_sum(sum1, dataset[rand() % (*num_primes)]);
            free_knot(sum1);
            sum1 = sum2;
        }
        dataset[i + (*num_primes)] = sum1;
        if(dataset[i]->num_crossings > max_crossings){
            max_crossings = dataset[i]->num_crossings;
        }
        if(dataset[i + (*num_primes)]->num_crossings > max_crossings){
            max_crossings = dataset[i + (*num_primes)]->num_crossings;
        }
    }
    max_crossings *= padding;
    // printf("%d\n", max_crossings);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    double time_avg = time_taken / total_additions;
    printf("Performed %d additions in %fs (on average %fs per addition).\n", total_additions, time_taken, time_avg);

    // printf("Max crossings: %d\n", max_crossings);

    start = clock();
    int num_moves = 0;
    for(int i = 0; i < 2 * (*num_primes); i++) {
        //printf("%d", i);
        Knot **alts = create_alters(dataset[i], num_alts, max_crossings + rand() % max_crossings + num_alts, &num_moves);

        if(num_alts > 0){
            dataset[i] = alts[0];
        }
        for(int j = 0; j < num_alts - 1; j++) {
            dataset[2 * (*num_primes) + i * (num_alts - 1) + j] = alts[j + 1];
            //printf(", %d", 2 * (*num_primes) + i * (num_alts - 1) + j);
        }
        free(alts);
        //printf("\n");
        // dataset[i] = increase_crossings(dataset[i], max_crossings + rand() % max_crossings + num_alts, &num_moves);
    }
    *num_primes = 2 * num_alts * (*num_primes);
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    time_avg = time_taken / num_moves;
    printf("Performed %d rm moves in %fs (on average %fs per move).\n", num_moves, time_taken, time_avg);

    return dataset;
}

//void de_flype(Knot *knot, int *flype_pairs){
//
//}
