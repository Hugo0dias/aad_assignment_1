// AVX-512 Minerador com OpenMP (mantém lógica original de aad_sha1_cpu_miner_avx512_Instr.c)
// Requer CPU com AVX-512 (f) e sha1_avx512f() disponível em aad_sha1_cpu.h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <immintrin.h>
#include <stdint.h>
#include <emmintrin.h>
#include <omp.h>

#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"

#define N_LANES 16
#define LANE_SIZE 56
#define NONCE_BYTES 42
#define NONCE_CHUNKS 5
#define NONCE_REM 2

typedef struct {
    __m512i chunks[NONCE_CHUNKS * 2];
    uint16_t rem[N_LANES];
} nonce16_t __attribute__((aligned(64)));

static inline void increment_nonces16(nonce16_t *nonce) {
    __m512i one = _mm512_set1_epi64(1);
    nonce->chunks[0] = _mm512_add_epi64(nonce->chunks[0], one);
    nonce->chunks[1] = _mm512_add_epi64(nonce->chunks[1], one);
}

static inline int prefix_matches_aad2025(uint32_t first_word) {
    return first_word == 0xAAD20250u;
}

int main(void) {
    const char template_str[12] = "DETI coin 2 ";
    const unsigned long long REPORT_INTERVAL = 100000000ULL;

    unsigned long long total_attempts = 0ULL;
    double global_start_time = omp_get_wtime();
    double last_report_time = global_start_time;

    printf("Starting AVX-512 miner with OpenMP (original logic)...\n");

    srand((unsigned int)time(NULL));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        unsigned long long n_attempts = 0ULL;

        uint8_t msg[N_LANES * LANE_SIZE] __attribute__((aligned(64)));
        nonce16_t nonce;
        v16si coin[14];
        v16si hash[5];
        uint32_t coin_lane[14];

        // Initialize msg com template + padding + nonces aleatórios
        for (int lane = 0; lane < N_LANES; lane++) {
            memcpy(msg + lane * LANE_SIZE, template_str, 12);
            msg[lane * LANE_SIZE + 54] = '\n';
            msg[lane * LANE_SIZE + 55] = 0x80;
            for (int j = 0; j < NONCE_BYTES; j++) {
                uint8_t byte;
                do {
                    byte = rand() & 0xFF;
                } while (byte == 0x0A || byte == 0x80);
                msg[lane * LANE_SIZE + 12 + j] = byte;
            }
        }

        // Main mining loop (MANTÉM LÓGICA ORIGINAL)
        while (1) {
            if (n_attempts == 0ULL) {
                // Primeira iteração: inicializa nonce a partir de msg
                for (int lane = 0; lane < N_LANES; lane++) {
                    uint64_t *p = (uint64_t*)(msg + lane * LANE_SIZE + 12);
                    for (int c = 0; c < NONCE_CHUNKS; c++) {
                        ((uint64_t*)&nonce.chunks[c])[lane] = p[c];
                    }
                    nonce.rem[lane] = ((uint16_t*)(&msg[lane * LANE_SIZE + 12 + NONCE_CHUNKS*8]))[0];
                }
            } else {
                // Próximas iterações: incrementa nonce
                increment_nonces16(&nonce);
                // Copia nonce actualizada para msg
                for (int lane = 0; lane < N_LANES; lane++) {
                    uint64_t *p = (uint64_t*)(msg + lane * LANE_SIZE + 12);
                    for (int c = 0; c < NONCE_CHUNKS; c++) {
                        int chunk_index = c * 2 + (lane >= 8);
                        int lane_index = lane % 8;
                        p[c] = ((uint64_t*)&nonce.chunks[chunk_index])[lane_index];
                    }
                    ((uint16_t*)(&msg[lane * LANE_SIZE + 12 + NONCE_CHUNKS*8]))[0] = nonce.rem[lane];
                }
            }

            n_attempts += N_LANES;

            // Monta coin_u32: cada word i é um vector com as 16 lanes
            uint32_t *coin_u32 = (uint32_t*)coin;
            for (int i = 0; i < 14; i++) {
                for (int lane = 0; lane < N_LANES; lane++) {
                    uint32_t w = *(uint32_t*)(msg + lane * LANE_SIZE + i * 4);
                    coin_u32[i * N_LANES + lane] = __builtin_bswap32(w);
                }
            }

            // Chama SHA1 AVX-512
            sha1_avx512f((v16si*)coin, (v16si*)hash);

            // Checa h0 por lane
            uint32_t *hash_u32 = (uint32_t*)hash;
            for (int lane = 0; lane < N_LANES; lane++) {
                uint32_t h0_of_lane = hash_u32[0 * N_LANES + lane];
                if (prefix_matches_aad2025(h0_of_lane)) {
                    for (int i = 0; i < 14; i++)
                        coin_lane[i] = coin_u32[i * N_LANES + lane];

                    #pragma omp critical
                    {
                        save_coin(coin_lane);
                        save_coin(NULL);
                        double elapsed = wall_time_delta();
                        if (elapsed > 0.0)
                            printf("%.3f Mhashes/s\n", (double)n_attempts / (elapsed * 1e6));
                        printf("\033[1;32mThread %d found DETI coin after %llu attempts (lane %d)!\033[0m\n",
                               tid, n_attempts + (unsigned long long)lane + 1ULL, lane);
                    }
                }
            }

            // Reporting every REPORT_INTERVAL
            if ((n_attempts % REPORT_INTERVAL) == 0) {
                unsigned long long prev_total;
                #pragma omp atomic capture
                prev_total = total_attempts += n_attempts;
                n_attempts = 0ULL;

                #pragma omp master
                {
                    double now = omp_get_wtime();
                    double elapsed = now - last_report_time;
                    double total_elapsed = now - global_start_time;
                    last_report_time = now;

                    if (elapsed > 0.0) {
                        double mhash_rate = (double)REPORT_INTERVAL / (elapsed * 1e6);
                        printf("Attempts: %llu  (%.2f Mhashes/s)  Total elapsed: %.1f sec\n",
                               prev_total, mhash_rate, total_elapsed);
                        fflush(stdout);
                    }
                }
            }
        }
    } // end parallel

    return 0;
}
