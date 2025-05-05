#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <time.h>

int main(int argc, char *argv[]) {
    int num_lines = 0;

    if(argc == 2){
        num_lines = (int)atoi(argv[1]);
    }

    Knot **entries = read_prime_knots("PD_3-16.txt", &num_lines);

    entries = create_dataset(entries, &num_lines, 1, 1);
    // printf("Number of entries: %d\n", num_lines);

    MedialGraph **mgs = getMedialGraphs(entries, num_lines);
    generate_ids(mgs, num_lines);

    write_to_file(mgs, num_lines, "MedialGraphs/medial_graphs.txt");

    free_knots(entries, num_lines);
    free_mgs(mgs, num_lines);

    return 0;
}
