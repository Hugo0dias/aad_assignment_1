// Versao 1 : 512 -> Manteve o tipo de dados em m128i porem o v16si para AVX-512, aumentou as lanes
// aad_sha1_cpu_miner_avx512.c
// Requer CPU com AVX-512 (f) e sha1_avx512f() disponível em aad_sha1_cpu.h


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <immintrin.h>
#include <stdint.h>
#include <emmintrin.h>  // usamos loadl/storel para 8-bytes chunks

#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"

#define N_LANES 8            // AVX-512: 512 bits / 32 bits = 16 lanes
#define LANE_SIZE 56
#define NONCE_BYTES 42

static inline int prefix_matches_aad2025(uint32_t first_word) {
    return first_word == 0xAAD20250u;
}

/*
 * Incrementa os nonces diretamente dentro de msg usando intrinsics (SSE2 64-bit chunks).
 * Mantemos a mesma estratégia segura de antes: fazemos add de 64-bit por bloco
 * e não deixamos que o carry escape de NONCE_BYTES.
 *
 * Cada lane está em msg[lane*LANE_SIZE + 12 ... + 53]
 */
#define N_LANES 8
#define NONCE_CHUNKS 5   // 5 chunks [0-7; 8-15; 16-23; 24-31; 32-39] do nonce
#define NONCE_REM 2      // bytes restantes [40-41] do nonce


typedef struct {
    __m256i chunks[NONCE_CHUNKS * 2];  // 2 porque cada __m256i tem 8 lanes de 32 bits e sao 16 lanes no total
    uint16_t rem[N_LANES];
} nonce16_t __attribute__((aligned(32)));

static inline void increment_nonces16(nonce16_t *nonce) {
    __m256i one = _mm256_set1_epi32(1);

    // incrementa as 16 lanes (duas metades de 8)
    nonce->chunks[0] = _mm256_add_epi32(nonce->chunks[0], one);
    nonce->chunks[1] = _mm256_add_epi32(nonce->chunks[1], one);
}


int main(void) {
    unsigned long long n_attempts = 0ULL;
    double elapsed;

    uint8_t msg[N_LANES * LANE_SIZE] __attribute__((aligned(32))); // alinhado para AVX-512
    v8si coin[14];   // 14 words, cada uma com 8 lanes de 32 bits
    v8si hash[5];
    uint32_t coin_lane[14];
    nonce16_t nonce;

    const char template[12] = "DETI coin 2 ";

    srand((unsigned int)time(NULL));

    // Inicializa template + padding + nonces aleatórios
    for (int lane = 0; lane < N_LANES; lane++) {
        memcpy(msg + lane * LANE_SIZE, template, 12);
        msg[lane * LANE_SIZE + 54] = '\n';
        msg[lane * LANE_SIZE + 55] = 0x80;
        for (int j = 0; j < NONCE_BYTES; j++) {
            uint8_t byte;
            do {
                byte = rand() & 0xFF;
            } while (byte == 0x0A || byte == 0x80);  // evita '\n' e padding
            msg[lane * LANE_SIZE + 12 + j] = byte;
        }
    }

    // Imprime msg em hexadecimal com endereços
    //printf("Initial msg (hexadecimal with addresses):\n");
    //for (int lane = 0; lane < N_LANES; lane++) {
    //    printf("Lane %2d [%p - %p]:\n", lane, 
    //           (void*)(msg + lane * LANE_SIZE), 
    //           (void*)(msg + lane * LANE_SIZE + LANE_SIZE - 1));
    //    for (int i = 0; i < LANE_SIZE; i++) {
    //        uint8_t *addr = msg + lane * LANE_SIZE + i;
    //        printf("  [%p] %02X ", (void*)addr, *addr);
    //        if ((i + 1) % 16 == 0) printf("\n");  // 16 bytes por linha
    //    }
    //    printf("\n");
    //}
    //printf("\n");

    time_measurement();

    while (1) {

        if (n_attempts == 0ULL) {
            for (int lane = 0; lane < N_LANES; lane++) {
                uint64_t *p = (uint64_t*)(msg + lane * LANE_SIZE + 12); // ponteiro para chunks do nonce
                for (int c = 0; c < NONCE_CHUNKS; c++) {
                    /*  
                        nonce.chunks[c]           → um registo __m512i (512 bits)
                        &nonce.chunks[c]          → endereço do registo __m512i
                        (uint64_t*)&nonce.chunks[c] → reinterpreta como array de uint64_t
                        ((uint64_t*)&nonce.chunks[c])[lane] → acede ao elemento [lane] do array uint64_t
                    */
                    ((uint64_t*)&nonce.chunks[c])[lane] = p[c]; // COPIA o nonce de msg para nonce16_t
                    // printf("Lane %2d nonce at start:\n", lane);
                    // printf("  Chunk %d: %016lX\n", c, p[c]);
                }
                nonce.rem[lane] = ((uint16_t*)(&msg[lane * LANE_SIZE + 12 + NONCE_CHUNKS*8]))[0]; // bytes restantes
                // printf("  Rem: %04X\n", nonce.rem[lane]);
            }
        } else {
            increment_nonces16(&nonce);
            // copia novos nonces para msg
            for (int lane = 0; lane < N_LANES; lane++) {
                uint64_t *p = (uint64_t*)(msg + lane * LANE_SIZE + 12); // ponteiro para chunks do nonce
                // printf("Lane %2d nonce after increment:\n", lane);
                for (int c = 0; c < NONCE_CHUNKS; c++) {
                    // escolhe 0/1 para lanes 0–7 ou 8–15 se lane >= 8, registos impares (7-15) e vice versa
                    int chunk_index = c * 2 + (lane >= 8);
                    // escolhe lane % 8 para lanes 0–7 ou 8–15
                    int lane_index  = lane % 8;
                    // 
                    p[c] = ((uint64_t*)&nonce.chunks[chunk_index])[lane_index];
                }
                ((uint16_t*)(&msg[lane * LANE_SIZE + 12 + NONCE_CHUNKS*8]))[0] = nonce.rem[lane];
                // printf("  Rem: %04X\n", nonce.rem[lane]);
            }

        }


        // monta coin_u32: cada word i é um vector com as 16 lanes
        // coin_u32[ i*N_LANES + lane ] = word (big-endian)
        uint32_t *coin_u32 = (uint32_t*)coin;
        for (int i = 0; i < 14; i++) {
            for (int lane = 0; lane < N_LANES; lane++) {
                // lê 4 bytes do msg (little-endian em memória), converte para big-endian word
                uint32_t w = *(uint32_t*)(msg + lane * LANE_SIZE + i * 4);
                coin_u32[i * N_LANES + lane] = __builtin_bswap32(w);
            }
        }

        // chama SHA1 AVX-2 (deve existir na tua lib aad_sha1_cpu.h)
        sha1_avx2((v8si*)coin, (v8si*)hash);

        // verifica h0 por lane
        uint32_t *hash_u32 = (uint32_t*)hash;
        for (int lane = 0; lane < N_LANES; lane++) {
            uint32_t h0_of_lane = hash_u32[0 * N_LANES + lane];
            if (h0_of_lane == 0xEB90509u) {
                printf("Found target hash in lane %d: %08X\n", lane, h0_of_lane);
            }
            // printf("Lane %2d: h0 = %08X\n", lane, h0_of_lane);
            if (prefix_matches_aad2025(h0_of_lane)) {
                for (int i = 0; i < 14; i++)
                    coin_lane[i] = coin_u32[i * N_LANES + lane];
                save_coin(coin_lane);
                save_coin(NULL);

                time_measurement();
                elapsed = wall_time_delta();
                if (elapsed > 0.0)
                    printf("%.3f Mhashes/s\n", (double)n_attempts / (elapsed * 1e6));
                printf("\033[1;32m Found DETI coin after %llu attempts (lane %d)!\033[0m\n",
                       n_attempts + (unsigned long long)lane + 1ULL, lane);
            }
        }

        n_attempts += N_LANES;
        if (n_attempts > 8996895643ULL + 1ULL)
            break;

        if (n_attempts % 100000000ULL == 0ULL) {
            time_measurement();
            elapsed = wall_time_delta();
            printf("Attempts: %llu\n", n_attempts);
            if (elapsed > 0.0)
                printf("%.4f Mhashes/s\n", (double)100000000ULL / (elapsed * 1e6));  // taxa de este período
        }
    }

    return 0;
}
