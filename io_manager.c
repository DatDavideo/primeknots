#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

#define MAX_LINE_LENGTH 1024

void write_to_file(MedialGraph **mgs, int num_graphs, const char *filename) {
    clock_t start = clock();

    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file");
        return;
    }
    for (int i = 0; i < num_graphs; i++) {
        fprintf(file, "{");
        fprintf(file, "\"num_faces\": %d, ", mgs[i]->num_faces);
        fprintf(file, "\"prime\": %d, ", mgs[i]->prime);
        fprintf(file, "\"id\": %d, ", mgs[i]->id);
        fprintf(file, "\"parents\": [");
        for(int j = 0; j < mgs[i]->num_parents - 1; j++){
            fprintf(file, "%d,", mgs[i]->parents[j]);
        }
        fprintf(file, "%d], ", mgs[i]->parents[mgs[i]->num_parents - 1]);
        fprintf(file, "\"source\": [");
        for(int j = 0; j < mgs[i]->num_edges - 1; j++){
            fprintf(file, "%d,", mgs[i]->source[j]);
        }
        fprintf(file, "%d], ", mgs[i]->source[mgs[i]->num_edges - 1]);
        fprintf(file, "\"target\": [");
        for(int j = 0; j < mgs[i]->num_edges - 1; j++){
            fprintf(file, "%d,", mgs[i]->target[j]);
        }
        fprintf(file, "%d], ", mgs[i]->target[mgs[i]->num_edges - 1]);

        fprintf(file, "\"sign\": [");
        for(int j = 0; j < mgs[i]->num_edges - 1; j++){
            fprintf(file, "%d,", mgs[i]->sign[j]);
        }
        fprintf(file, "%d]", mgs[i]->sign[mgs[i]->num_edges - 1]);
        fprintf(file, "}\n");
    }
    fclose(file);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    double time_per_graph = time_taken / num_graphs;
    printf("Wrote %d entries in %fs (on average %fs per entry).\n", num_graphs, time_taken, time_per_graph);
}

int is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int next_prime(int n) {
    int candidate = n + 1;
    while (!is_prime(candidate)) {
        candidate++;
    }
    return candidate;
}

int count_lines(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return 0;
    }
    int line_count = 0;
    char buffer[8192];
    while (!feof(file)) {
        size_t bytes_read = fread(buffer, 1, sizeof(buffer), file);
        for (size_t i = 0; i < bytes_read; i++) {
            if (buffer[i] == '\n') {
                line_count++;
            }
        }
    }
    fclose(file);
    return line_count;
}

Knot *create_entry(int id, char *pd_diagram, int prime) {
    Knot *entry = malloc(sizeof(Knot));
    entry->prime = prime;
    entry->parents = malloc(sizeof(int));
    entry->parents[0] = id;
    entry->num_parents = 1;

    int num_crossings = 0;
    const char *comma_ctr = pd_diagram;
    while (*comma_ctr != '\0') {
        if(*comma_ctr == ','){
            num_crossings++;
        }
        comma_ctr++;
    }
    num_crossings = (num_crossings + 1) / 4;
    entry->num_crossings = num_crossings;
    entry->crossings = malloc((size_t)num_crossings * 4 * sizeof(int));

    const char *ptr = pd_diagram;
    int ctr = 0;
    while (*ptr != '\0') {
        if (isdigit(*ptr)) {
            int num = 0;
            while (isdigit(*ptr)) {
                num = num * 10 + (*ptr - '0');
                ptr++;
            }
            entry->crossings[ctr] = num;
            ctr++;
        } else {
            ptr++;
        }
    }
    for(int i = 0; i < num_crossings * 4; i++) {
        if(entry->crossings[i] == 0){
            entry->crossings[i] = num_crossings * 2;
        }
    }
    return entry;
}

Knot **read_prime_knots(const char *filename, int *num_lines){
    clock_t start = clock();

    if(*num_lines == 0){
        *num_lines = count_lines(filename);
    }
    Knot **entries = malloc((size_t)*num_lines * sizeof(Knot *));
    if (!entries) {
        perror("malloc err");
        return NULL;
    }
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    int count = 0;
    int id = 2;

    while (fgets(line, sizeof(line), file) && count < *num_lines) {
        strtok(line, ",");
        strtok(NULL, ",");

        char *pd_diagram = strtok(NULL, "\n");
        entries[count] = create_entry(id, pd_diagram, 1);
        id = next_prime(id);
        // id++;

        count++;
    }

    fclose(file);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    double time_per_knot = time_taken / *num_lines;
    printf("Read %d entries in %fs (on average %fs per entry).\n", *num_lines, time_taken, time_per_knot);

    return entries;
}
