#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"
#include <omp.h>

#define MIN_CHAR 0x00
#define MAX_CHAR 0xFF

int main(int argc, char *argv[])
{
    double time_limit = 0.0;
    const char *name = NULL;
    unsigned long long thread_attempts[128] = {0};

    if (argc > 1) {
        time_limit = atof(argv[1]);
        if (time_limit <= 0.0) {
            printf("Invalid time value. Usage: %s [time_in_seconds] [name]\n", argv[0]);
            return 1;
        }
        printf("Running for %.2f seconds...\n", time_limit);
    } else {
        printf("No time limit — mining inf.\n");
    }

    if (argc > 2) {
        name = argv[2];
        printf("Searching coins with embedded name: \"%s\"\n", name);
    }

    size_t name_len = name ? strlen(name) : 0;
    if (name_len > 42) name_len = 42;

    const char template[12] = "DETI coin 2 ";
    double start_time = omp_get_wtime();
    unsigned long long n_attempts_total = 0ULL;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        unsigned long long n_attempts = 0ULL;

        u32_t coin[14];
        u32_t hash[5];
        u08_t *msg = (u08_t *)coin;

        // Inicializa template + nome
        for (int i = 0; i < 12; i++)
            msg[i ^ 3] = (u08_t)template[i];
        msg[54 ^ 3] = '\n';
        msg[55 ^ 3] = 0x80;
        if (name != NULL) {
            for (size_t i = 0; i < name_len; i++)
                msg[(12 + i) ^ 3] = (u08_t)name[i];
        }

        // Calcula faixa exclusiva do primeiro byte do nonce para cada thread
        int range_size = (MAX_CHAR - MIN_CHAR + 1) / num_threads;
        int start_byte = MIN_CHAR + tid * range_size;
        int end_byte = (tid == num_threads - 1) ? MAX_CHAR : start_byte + range_size - 1;

        // Inicializa todo o nonce (bytes 12+name_len .. 53) com o mínimo da faixa
        int nonce_start = 12 + name_len;
        for (int i = nonce_start; i < 54; i++)
            msg[i ^ 3] = MIN_CHAR;

        // Define primeiro byte do nonce como identificador da thread
        msg[nonce_start ^ 3] = (u08_t)start_byte;

        while (1) {
            // Incremento sequencial do odômetro (mantendo lógica original)
            int carry = 1; // queremos incrementar pelo menos 1
            for (int i = 53; i > nonce_start; i--) { // i > nonce_start para não tocar no thread-ID
                if (carry) {
                    msg[i ^ 3]++;
                
                    if (msg[i ^ 3] == 0x0A)  // pula o \n
                        msg[i ^ 3]++;
                
                    if (msg[i ^ 3] == 0x00) // overflow, propaga carry
                        carry = 1;
                    else
                        carry = 0;
                }
            }

            sha1(coin, hash);
            n_attempts++;

            if (hash[0] == 0xAAD20250u) {
                #pragma omp critical
                {
                    printf("\033[1;32mThread %d found DETI coin after %llu attempts!\033[0m\n",
                           tid, n_attempts_total + n_attempts);
                    save_coin(coin);
                    save_coin(NULL);
                }
            }

            double now = omp_get_wtime();
            double elapsed_local = now - start_time;
            if (time_limit > 0.0 && elapsed_local >= time_limit) {
                thread_attempts[tid] += n_attempts;
                #pragma omp barrier
                #pragma omp master
                {
                    n_attempts_total = 0ULL;
                    int nt = omp_get_num_threads();
                    for (int t = 0; t < nt; t++) n_attempts_total += thread_attempts[t];
                    double final_elapsed = omp_get_wtime() - start_time;
                    printf("\n=== Miner stopped ===\n");
                    printf("Total attempts: %llu\n", n_attempts_total);
                    printf("Elapsed time: %.2f seconds\n", final_elapsed);
                    printf("Average rate: %.2f Mhash/s\n", (double)n_attempts_total / final_elapsed / 1e6);
                }
                break;
            }
        }
    }

    return 0;
}
