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
#define NONCE_CHUNKS 4
#define NONCE_REM 2

int count = 0;

typedef struct {
    __m512i chunks[NONCE_CHUNKS * 2]; 
    __m512i chunks_high[NONCE_CHUNKS / 2]; // chunk 0 e 1 fixos para threads e lanes
    uint16_t rem[N_LANES];
} nonce16_t __attribute__((aligned(64)));


static inline void print_m512i(__m512i v) {
    uint64_t vals[8];
    _mm512_storeu_si512(vals, v);

    for (int i = 0; i < 8; i++) {
        printf("%016lx ", vals[i]);
    }
    printf("\n");
}

static inline void increment_nonces16(nonce16_t *nonce) {

    const __m512i all_ones = _mm512_set1_epi64((long long) -1); /* 0xFFFFFFFFFFFFFFFFULL */

    /* compara chunk[count] e chunk[count+1] com all_ones (element-wise)
       _mm512_cmpeq_epi64_mask devolve máscara de 8 bits (1 bit por lane). */
    __mmask8 m0 = _mm512_cmpeq_epi64_mask(nonce->chunks[count], all_ones);
    __mmask8 m1 = _mm512_cmpeq_epi64_mask(nonce->chunks[count + 1], all_ones);

    //printf("Masks for chunks %d and %d: m0=0x%02X, m1=0x%02X\n", count, count + 1, m0, m1);

    /* se ambos forem todas as lanes == all_ones, avança count em 2 (procura próximo par livre) */
    const __mmask8 all_mask = (1u << 8) - 1; /* 0xFF para 8 lanes */
    if (m0 == all_mask && m1 == all_mask) {
        printf("All lanes exhausted for chunks %d and %d, moving to next pair.\n", count, count + 1);
        count += 2;
        if (count >= sizeof(nonce->chunks) - 1) count = 0; /* wrap-around seguro */
    }

    __m512i one = _mm512_set1_epi64(1);

    // printf("Incrementing nonces...\n");
// 
    // printf("Before increment (chunk %d):\n", count);
    // print_m512i(nonce->chunks_high[0]);
    // printf("chunk %d):\n", count+2);
    // print_m512i(nonce->chunks[count]);
    // printf("chunk %d):\n", count+4);
    // print_m512i(nonce->chunks[count + 2]);
    // printf("chunk %d):\n", count+6);
    // print_m512i(nonce->chunks[count + 4]);


    nonce->chunks[count] = _mm512_add_epi64(nonce->chunks[count], one);
    nonce->chunks[count + 1] = _mm512_add_epi64(nonce->chunks[count + 1], one);

    // printf("After increment (chunk %d):\n", count);
    // print_m512i(nonce->chunks_high[0]);
    // printf("chunk %d):\n", count+2);
    // print_m512i(nonce->chunks[count]);
    // printf("chunk %d):\n", count+4);
    // print_m512i(nonce->chunks[count + 2]);
    // printf("chunk %d):\n", count+6);
    // print_m512i(nonce->chunks[count + 4]);

}


static inline int prefix_matches_aad2025(uint32_t first_word) {
    return first_word == 0xAAD20250u;
}

int main(int argc, char *argv[]) {
    double elapsed = 0.0;
    double time_limit = 0.0;
    const char *name = NULL;
    unsigned long long thread_attempts[128] = {0}; // suporte até 128 threads

    if (argc > 1) {
        time_limit = atof(argv[1]);
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

    size_t name_len = name ? strlen(name) : 0;
    if (name_len > 42) name_len = 42;

    const char template[12] = "DETI coin 2 ";
    unsigned long long n_attempts_total = 0ULL;
    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned long long n_attempts = 0ULL;
        int myid = tid;  // alias simples

        uint8_t msg[N_LANES * LANE_SIZE] __attribute__((aligned(64)));
        nonce16_t nonce;
        v16si coin[14];
        v16si hash[5];
        uint32_t coin_lane[14];

        for (int lane = 0; lane < N_LANES; lane++) {
            memcpy(msg + lane * LANE_SIZE, template, 12);
            msg[lane * LANE_SIZE + 54] = '\n';
            msg[lane * LANE_SIZE + 55] = 0x80;
            msg[lane * LANE_SIZE + 53] = 0x00;
            msg[lane * LANE_SIZE + 52] = 0x00;
        }

        // Inicializa chunks_high[0] com valores únicos para lanes 0-7
        uint64_t base_nonces_low[8];
        for (int lane = 0; lane < 8; lane++) {
            base_nonces_low[lane] = ((uint64_t)tid << 48) | ((uint64_t)lane << 42);
            // printf("Thread %d Lane %d base_nonce: 0x%016lX\n", tid, lane, base_nonces_low[lane]);
        }
        nonce.chunks_high[0] = _mm512_setr_epi64(
            base_nonces_low[0], base_nonces_low[1], base_nonces_low[2], base_nonces_low[3],
            base_nonces_low[4], base_nonces_low[5], base_nonces_low[6], base_nonces_low[7]
        );

        // Inicializa chunks_high[1] com valores únicos para lanes 8-15
        uint64_t base_nonces_high[8];
        for (int lane = 8; lane < 16; lane++) {
            base_nonces_high[lane - 8] = ((uint64_t)tid << 48) | ((uint64_t)lane << 42);
            // printf("Thread %d Lane %d base_nonce: 0x%016lX\n", tid, lane, base_nonces_high[lane - 8]);
        }
        nonce.chunks_high[1] = _mm512_setr_epi64(
            base_nonces_high[0], base_nonces_high[1], base_nonces_high[2], base_nonces_high[3],
            base_nonces_high[4], base_nonces_high[5], base_nonces_high[6], base_nonces_high[7]
        );     

        // Seed diferente por thread
        unsigned int seed = (unsigned int)(time(NULL) ^ (tid << 16) ^ (uintptr_t)&nonce);

        // Inicializa chunks[] com valores aleatórios
        for (int i = 0; i < NONCE_CHUNKS * 2; i++) {
            uint64_t random_values[8];
            for (int lane = 0; lane < 8; lane++) {
                // Gera 64 bits aleatórios
                uint64_t r1 = (uint64_t)rand_r(&seed);
                uint64_t r2 = (uint64_t)rand_r(&seed);
                random_values[lane] = (r1 << 32) | r2;
            }
            nonce.chunks[i] = _mm512_setr_epi64(
                random_values[0], random_values[1], random_values[2], random_values[3],
                random_values[4], random_values[5], random_values[6], random_values[7]
            );
        }

        
        // Main mining loop (MANTÉM LÓGICA ORIGINAL)
        while (1) {
            
            increment_nonces16(&nonce);

            const __m512i scatter_indices = _mm512_set_epi64(
                7 * LANE_SIZE,  // lane 7
                6 * LANE_SIZE,  // lane 6
                5 * LANE_SIZE,  // lane 5
                4 * LANE_SIZE,  // lane 4
                3 * LANE_SIZE,  // lane 3
                2 * LANE_SIZE,  // lane 2
                1 * LANE_SIZE,  // lane 1
                0 * LANE_SIZE   // lane 0
            );

            // ========== COPIA CHUNKS_HIGH (primeiros 8 bytes do nonce) ==========
            // Bytes 12-19 de cada lane
            _mm512_i64scatter_epi64(
                msg + 12,              // base address (offset 12 = após "DETI coin 2 ")
                scatter_indices,
                nonce.chunks_high[0],  // lanes 0-7
                1
            );

            _mm512_i64scatter_epi64(
                msg + 8 * LANE_SIZE + 12,  // base para lanes 8-15
                scatter_indices,
                nonce.chunks_high[1],      // lanes 8-15
                1
            );

            // ========== COPIA CHUNKS (próximos 32 bytes = 4 chunks) ==========
            // Chunk 0 (bytes 20-27)
            _mm512_i64scatter_epi64(
                msg + 12 + 8,          // offset 20
                scatter_indices,
                nonce.chunks[0],       // lanes 0-7
                1
            );
            _mm512_i64scatter_epi64(
                msg + 8 * LANE_SIZE + 12 + 8,
                scatter_indices,
                nonce.chunks[1],       // lanes 8-15
                1
            );

            // Chunk 1 (bytes 28-35)
            _mm512_i64scatter_epi64(
                msg + 12 + 16,
                scatter_indices,
                nonce.chunks[2],       // lanes 0-7
                1
            );
            _mm512_i64scatter_epi64(
                msg + 8 * LANE_SIZE + 12 + 16,
                scatter_indices,
                nonce.chunks[3],       // lanes 8-15
                1
            );

            // Chunk 2 (bytes 36-43)
            _mm512_i64scatter_epi64(
                msg + 12 + 24,
                scatter_indices,
                nonce.chunks[4],       // lanes 0-7
                1
            );
            _mm512_i64scatter_epi64(
                msg + 8 * LANE_SIZE + 12 + 24,
                scatter_indices,
                nonce.chunks[5],       // lanes 8-15
                1
            );

            // Chunk 3 (bytes 44-51)
            _mm512_i64scatter_epi64(
                msg + 12 + 32,
                scatter_indices,
                nonce.chunks[6],       // lanes 0-7
                1
            );
            _mm512_i64scatter_epi64(
                msg + 8 * LANE_SIZE + 12 + 32,
                scatter_indices,
                nonce.chunks[7],       // lanes 8-15
                1
            );

            // ========== COPIA REMAINDER (últimos 2 bytes, bytes 52-53) ==========
            for (int lane = 0; lane < N_LANES; lane++) {
                *(uint16_t*)(msg + lane * LANE_SIZE + 12 + 40) = nonce.rem[lane];
            }

            n_attempts += N_LANES;

            // Monta coin_u32: cada word i é um vector com as 16 lanes
            uint32_t *coin_u32 = (uint32_t*)coin;
            
            
            for (int i = 0; i < 14; i++) {
                for (int lane = 0; lane < N_LANES; lane++) {
                    uint32_t w = *(uint32_t*)(msg + lane * LANE_SIZE + i * 4);
                    coin_u32[i * N_LANES + lane] = __builtin_bswap32(w);
                    //printf("Thread %d Coin word %d lane %d: 0x%08X\n", tid, i, lane, coin_u32[i * N_LANES + lane]);
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

            double now = omp_get_wtime();
            double elapsed_local = now - start_time;

            if (time_limit > 0.0 && elapsed_local >= time_limit) {
                /* guarda contagens locais numa slot partilhada para agregação */
                thread_attempts[myid] += n_attempts;
                n_attempts = 0ULL;
                        
                /* sincronizar todas as threads: damos tempo para as outras escreverem a sua contagem */
                #pragma omp barrier
                        
                /* apenas a master agrega e imprime */
                #pragma omp master
                {
                    n_attempts_total = 0ULL;
                    int nt = omp_get_num_threads();
                    for (int t = 0; t < nt; t++)
                        n_attempts_total += thread_attempts[t];
                
                    double final_elapsed = omp_get_wtime() - start_time;
                    printf("=== Miner stopped ===\n");
                    printf("Total attempts: %llu\n", n_attempts_total);
                    printf("Elapsed time: %.2f seconds\n", final_elapsed);
                    printf("Average rate: %.2f Mhashes/s\n", (double)n_attempts_total / final_elapsed / 1e6);
                    printf("=== Tempo limite atingido ===\n");
                }
            
                /* garante que a master já fez a impressão antes de sair */
                #pragma omp barrier
            
                /* pede a paragem a todas as threads */
                //atomic_store(&stop_flag, 1);
            
                /* sai do laço (cada thread) */
                break;
            }
        }

    }
            return 0;

} // end parallel

