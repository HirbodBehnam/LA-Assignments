#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CROSS_TESTS 500

int rand_bool(int chance_in) {
    return rand() % chance_in == 0;
}

void generate_vector(FILE *file, int size) {
    while (size--)
        fprintf(file, " %d", rand() % 10);
}

int main() {
    srand(time(NULL));
    FILE *file = fopen("in.txt", "w");
    int q = rand() % 1000 + 1000;
    int n = rand() % 10 + 10;
    fprintf(file, "%d\n%d\n", n, q + CROSS_TESTS);
    while (q--) {
        switch (rand() % 7) {
            case 0:
                fprintf(file, "T");
                break;
            case 1:
                fprintf(file, "dot");
                generate_vector(file, rand_bool(3) ? n : (rand() % 20));
                break;
            case 2:
                fprintf(file, "out");
                generate_vector(file, rand_bool(2) ? n : (rand() % 20));
                fprintf(file, " , %d", (rand() % n) * 3 / 2);
                break;
            case 3:
                fprintf(file, "cross");
                generate_vector(file, rand_bool(2) ? 3 : (rand() % 10));
                break;
            case 4:
                fprintf(file, "had");
                generate_vector(file, rand_bool(2) ? n : (rand() % 20));
                break;
            case 5:
                fprintf(file, "print");
                break;
            case 6:
                n = rand() % 10 + 10;
                fprintf(file, "reset %d", n);
                break;
        }
        fprintf(file, "\n");
    }
    // Cross tests
    fputs("reset 3\n", file);
    for (int i = 1; i < CROSS_TESTS; i++) {
        switch (rand() % 4) {
            case 0:
                fprintf(file, "T");
                break;
            case 1:
                fprintf(file, "cross");
                generate_vector(file, rand_bool(2) ? 3 : (rand() % 10));
                break;
            case 2:
                fprintf(file, "had");
                generate_vector(file, 3);
                break;
            case 3:
                fprintf(file, "print");
                break;
        }
        fprintf(file, "\n");
    }
    fclose(file);
}
