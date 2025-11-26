#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"
#include <immintrin.h>
#include <omp.h>

#define N_LANES 8
#define COIN_SIZE 56 
#define BLOCK_SIZE 5

static inline int prefix_matches_aad2025(u32_t first_word) {
    return first_word == 0xAAD20250u; // cheque exacto do word H0 (mais seguro)
}

int main(int argc, char *argv[])
{
    double time_limit = 0.0; // 0 = sem limite
    const char *name = NULL;

    if (argc > 1) {
        time_limit = atof(argv[1]); // converte string -> double
        if (time_limit <= 0.0) {
            printf("Invalid time value. Usage: %s [time_in_seconds]\n", argv[0]);
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

    const char template[12] = "DETI coin 2 ";
    size_t name_len = name ? strlen(name) : 0;
    if(name_len > 42) name_len = 42;

    unsigned long long total_attempts = 0ULL;
    double start_time = omp_get_wtime();

    /* time measurement helpers (usa as tuas funções existentes) */
    time_measurement();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /* cada thread corre um motor idêntico com N_LANES lanes (vetorizadas) */
        unsigned long long n_attempts = 0ULL;

        u08_t msg[N_LANES * COIN_SIZE];
        v8si coin[14];
        v8si hash[5];

        /* inicializa o buffer msg por lane tal como no teu código original */
        for (int lane = 0; lane < N_LANES; lane++) {
            int base = lane * COIN_SIZE;
            /* template bytes 0..11 */
            for (int i = 0; i < 12; i++)
                msg[base + i] = (u08_t)template[i];
            /* nome (se houver) */
            if (name_len)
                for (size_t i = 0; i < name_len; i++)
                    msg[base + 12 + i] = (u08_t)name[i];
            /* restantes bytes do nonce inicializados a 0 */
            for (int i = 0; i < 42 - (int)name_len; i++)
                msg[base + 12 + name_len + i] = 0;
            msg[base + 54] = '\n';
            msg[base + 55] = 0x80;
        }

        /* iter conta quantos blocos/iterações já fez a thread */
        unsigned long long iter = 0ULL;

        /* loop principal por thread */
        while (1) {

            /* para cada lane, vamos gerar um final_nonce único por (iter,lane,tid) */
            for (int lane = 0; lane < N_LANES; lane++) {
                /* fórmula que garante partição entre threads:
                   final_nonce = ((iter * N_LANES) + lane) * nthreads + tid
                   isto faz com que nonces gerados por threads diferentes estejam em classes
                   congruentes diferentes mod nthreads -> não há colisões. */
                unsigned long long final_nonce = (((unsigned long long)iter * (unsigned long long)N_LANES)
                                                  + (unsigned long long)lane) * (unsigned long long)nthreads
                                                 + (unsigned long long)tid;

                /* escreve final_nonce little-endian nos bytes do nonce (mantendo teu código) */
                unsigned long long t = final_nonce;
                for (int i = 0; i < 42 - (int)name_len; i++) {
                    u08_t b = (u08_t)(t & 0xFFu);
                    t >>= 8;
                    if (b == 0x0A) b = 0x08; /* evita newline */
                    msg[lane*COIN_SIZE + 12 + name_len + i] = b;
                }
                /* garante padding consistentemente */
                msg[lane*COIN_SIZE + 54] = '\n';
                msg[lane*COIN_SIZE + 55] = 0x80;
            }

            /* monta palavras big-endian por lane (igual ao teu código) */
            for (int w = 0; w < 14; w++) {
                for (int lane = 0; lane < N_LANES; lane++) {
                    u32_t word = 0;
                    for (int b = 0; b < 4; b++) {
                        int idx = w * 4 + b;
                        if (idx < COIN_SIZE)
                            word |= ((u32_t)msg[lane*COIN_SIZE + (idx ^ 3)]) << (8 * b);
                    }
                    coin[w][lane] = word;
                }
            }

            /* chama AVX2 sha1 (mesmo nome e assinatura que tens) */
            sha1_avx2((v8si*)coin, (v8si*)hash);


            #pragma omp critical
            {
                printf("Thread %d: Trying nonce bytes:\n", tid);
                for (int lane = 0; lane < N_LANES; lane++) {
                    printf(" Lane %2d: ", lane);
                    for (int i = 0; i < 42; i++) {
                        printf("%02X ", msg[12 + lane*COIN_SIZE + (name_len + i) ^ 3]);
                    }
                    printf("\n");
                }
            }

            /* verifica resultados por lane, igual ao teu código */
            u32_t *hash_u32 = (u32_t*)hash;
            for (int lane = 0; lane < N_LANES; lane++) {
                u32_t h0_of_lane = hash_u32[0 * N_LANES + lane];
                if (prefix_matches_aad2025(h0_of_lane)) {
                    u32_t coin_lane_arr[14];
                    for (int w=0; w<14; w++) coin_lane_arr[w] = coin[w][lane];

                    #pragma omp critical
                    {
                        save_coin(coin_lane_arr);
                        save_coin(NULL);
                        printf("\033[1;32m[tid %d] Found DETI coin! (iter=%llu lane=%d)\033[0m\n", tid, iter, lane);
                    }
                }
            }

            n_attempts += N_LANES;

            /* incrementa iter para o próximo bloco de N_LANES nonces desta thread */
            iter++;

            /* time check (cada thread usa o mesmo start_time) */
            static const unsigned long long CHECK_FREQ = 1ULL << 15; /* verifica tempo periodicamente */
            if ((n_attempts & (CHECK_FREQ - 1)) == 0) {
                double now = omp_get_wtime();
                double global_elapsed = now - start_time;
                if (time_limit > 0.0 && global_elapsed >= time_limit) {
                    /* acumula e imprime (apenas a master) */
                    #pragma omp critical
                    {
                        total_attempts += n_attempts;
                    }
                    #pragma omp barrier
                    #pragma omp master
                    {
                        unsigned long long sum = total_attempts;
                        double final_elapsed = omp_get_wtime() - start_time;
                        printf("\n=== Miner stopped ===\n");
                        printf("Total attempts (aprox): %llu\n", sum);
                        printf("Elapsed time: %.2f seconds\n", final_elapsed);
                        printf("Average rate: %.2f Mhashes/s\n", (double)sum / final_elapsed / 1e6);
                        printf("=== Tempo limite atingido ===\n");
                    }
                    break;
                }
            }
        } /* end while thread */

        /* antes de terminar, acumula o que cada thread fez (protege) */
        #pragma omp critical
        {
            total_attempts += n_attempts;
        }
    } /* fim parallel */

    return 0;
}
