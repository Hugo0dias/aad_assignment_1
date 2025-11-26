#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"
#include <immintrin.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    double time_limit = 0.0; // 0 = sem limite
    const char *name = NULL;
    unsigned long long thread_attempts[128] = {0}; // suporte até 128 threads


    if (argc > 1) {
        time_limit = atof(argv[1]); // converte string -> double
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


    printf("Starting CPU miner (without SIMD)...\n");


    #define MIN_CHAR 0x20
    #define MAX_CHAR 0x9F

    size_t name_len = 0;

    const char template[12] = "DETI coin 2 ";
    double start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        unsigned long long n_attempts = 0ULL;
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int myid = tid;

        int nonce_bytes = 42 - (int)name_len;
        unsigned int indices_local[42];

        /* calcula faixa exclusiva do byte 0 para esta thread */
        int range_size = (MAX_CHAR - MIN_CHAR + 1) / num_threads;
        int start_byte = MIN_CHAR + tid * range_size;
        int end_byte   = start_byte + range_size - 1;
        if (tid == num_threads - 1) end_byte = MAX_CHAR; // resto para a última thread

        /* inicializa odómetro: todos 0 excepto o byte0 que fica na faixa da thread */
        for (int i = 0; i < nonce_bytes; ++i) indices_local[i] = 0;
        indices_local[0] = (unsigned int)(start_byte - MIN_CHAR); // STORE offset

        /* Cada thread tem seu próprio coin, hash, msg */
        u32_t coin[14];
        u32_t hash[5];
        u08_t *msg = (u08_t *)coin;

        /* monta template (cada thread uma cópia) */
        for (int i = 0; i < 12; i++) msg[i ^ 3] = (u08_t)template[i];
        msg[54 ^ 3] = '\n';
        msg[55 ^ 3] = 0x80;
        if (name != NULL) {
            for (size_t i = 0; i < name_len; i++)
                msg[(12 + i) ^ 3] = (u08_t)name[i];
        }

        while(1) {
            /* preenche msg com o nonce atual (NOTE: i==0 => byte 0 da faixa exclusivos da thread) */
            for (int i = 0; i < nonce_bytes; ++i) {
                msg[(12 + name_len + i) ^ 3] = (u08_t)(MIN_CHAR + indices_local[i]);
            }

            /* calcula hash */
            sha1(coin, hash);
            ++n_attempts;

            if (hash[0] == 0xAAD20250u) {
                #pragma omp critical
                {
                    printf("\033[1;32m Found DETI coin (thread %d) after %llu attempts!\033[0m\n", tid, n_attempts);
                    save_coin(coin);
                    save_coin(NULL);
                }
            }

            /* INCREMENTA odómetro: começa no byte menos significativo (último) */
            int carry = 1;
            for (int i = nonce_bytes - 1; i >= 0 && carry; --i) {
                /* se for o primeiro byte (i==0) tratamos a faixa especial */
                if (i == 0) {
                    unsigned int newv = indices_local[0] + 1;
                    if ((int)(MIN_CHAR + newv) > end_byte) {
                        /* volta ao início da faixa do thread e carry continua */
                        indices_local[0] = (unsigned int)(start_byte - MIN_CHAR);
                        carry = 1; /* carry tenta propagar para byte -1 (ignorado) */
                        /* não há byte -1: o odómetro deu wrap completo da faixa desta thread */
                    } else {
                        indices_local[0] = newv;
                        carry = 0;
                    }
                } else {
                    /* byte normal (pode ir 0..(MAX_CHAR-MIN_CHAR)) */
                    unsigned int newv = indices_local[i] + 1;
                    if (newv > (unsigned int)(MAX_CHAR - MIN_CHAR)) {
                        indices_local[i] = 0;
                        carry = 1;
                    } else {
                        indices_local[i] = newv;
                        carry = 0;
                    }
                }
            }

            /* (opcional) condição de paragem pelo tempo — mantém a tua lógica atual */
            double now = omp_get_wtime();
            double elapsed_local = now - start_time;
            if (time_limit > 0.0 && elapsed_local >= time_limit) {
                thread_attempts[myid] += n_attempts;
                n_attempts = 0ULL;
                #pragma omp barrier
                #pragma omp master
                {
                    unsigned long long sum = 0;
                    int nt = omp_get_num_threads();
                    for (int t = 0; t < nt; ++t) sum += thread_attempts[t];
                    double final_elapsed = omp_get_wtime() - start_time;
                    printf("=== Miner stopped ===\n");
                    printf("Total attempts: %llu\n", sum);
                    printf("Elapsed time: %.2f seconds\n", final_elapsed);
                    printf("Average rate: %.2f Mhashes/s\n", (double)sum / final_elapsed / 1e6);
                    printf("=== Tempo limite atingido ===\n");
                }
                #pragma omp barrier
                break;
            }
        } /* while */
    } /* parallel */
    return 0;
}
