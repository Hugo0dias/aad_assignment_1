#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"
#include <immintrin.h>

#define N_LANES 4

static inline int prefix_matches_aad2025(u32_t first_word) {
    return first_word == 0xAAD20250u; // cheque exacto do word H0 (mais seguro)
}

int main(int argc, char *argv[])
{
    double time_limit = 0.0; // 0 = sem limite

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

    unsigned long long n_attempts = 0ULL;

    u08_t msg[N_LANES * 56]; // N_LANES mensagens de 56 bytes cada
    v4si coin[14];           // cada element é um vector de 4 lanes (u32) -> 14*4bytes = 56 bytes * 4 = 224 Bytes
    v4si hash[5];

    // reserva 1os 12 bytes para o cabeçalho fixo da moeda, para cada uma das lanes
    const char template[12] = "DETI coin 2 ";
    for(int lane=0; lane<N_LANES; lane++) {
        for(int i=0;i<12;i++)
            msg[lane*56 + (i^3)] = (u08_t)template[i];
        msg[lane*56 + (54^3)] = '\n';
        msg[lane*56 + (55^3)] = 0x80;
    }

    // ponteiros para nonces (bytes 12..53)
    u08_t *nonce[N_LANES];
    for(int lane=0; lane<N_LANES; lane++)
        nonce[lane] = &msg[lane*56 + 12];

    // coin_lane = 1 coin temp
    u32_t coin_lane[14];
    
    double elapsed = 0.0;
    time_measurement();

    while(1) {
        // incrementa nonces (cada lane independentemente)
        for(int lane=0; lane<N_LANES; lane++) {
            // 53 - 12 + 1 = 42 bytes de nonce
            for(int i=0;i<42;i++) {
                if (++nonce[lane][i] == '\n')  // se for 0x0A, incrementa again
                    ++nonce[lane][i];
                if (nonce[lane][i] != 0) break; // carry
            }
        }

        
        u32_t *coin_u32 = (u32_t*)coin;
        u32_t *msg_u32  = (u32_t*)msg;
        for(int i=0;i<14;i++) {
            // Cada posição i contém agora 4 words (1 por lane)
            for(int lane=0; lane<N_LANES; lane++) {
                // msg = [word_0][lane_0], [word_1][lane_0], [word_2][lane_0], ...
                // msg estao todas por ordem de lane
                coin_u32[i * N_LANES + lane] = msg_u32[lane * 14 + i];
                // coin = [word_0][lane_0], [word_0][lane_1], [word_0][lane_2], ...
                // coin estao todas por ordem de word (1o tempos word 0 de todas as lanes, depois word 1 de todas as lanes, ...)
            }
        }

        // chama a versão AVX da sha1 que produz 4 hashes em paralelo
        sha1_avx((v4si*)coin, (v4si*)hash);

        // layout: ((u32_t*)hash) = [h0_l0,h0_l1,h0_l2,h0_l3, h1_l0,h1_l1,...]
        u32_t *hash_u32 = (u32_t*)hash;
        for(int lane=0; lane<N_LANES; lane++) {
            u32_t h0_of_lane = hash_u32[0 * N_LANES + lane]; // índice = word_index*4 + lane
            if(prefix_matches_aad2025(h0_of_lane)) {
                // extrair a coin dessa lane (reconstruir 14 words)
                for(int i=0;i<14;i++)
                    coin_lane[i] = coin_u32[i * N_LANES + lane];

                save_coin(coin_lane);
                printf("\033[1;32m Found DETI coin after %llu attempts!\033[0m\n", n_attempts);
		        save_coin(NULL);
            }
        }

        n_attempts += N_LANES;

        time_measurement();                 // mede nova amostra
        elapsed += wall_time_delta();       // acumula tempo decorrido

        if (time_limit > 0.0 && elapsed >= time_limit) {
            printf("\n=== Tempo limite atingido (%.2f / %.2f) ===\n", elapsed, time_limit);
            printf("Tentativas totais: %llu\n", n_attempts);
            printf("Taxa média: %.2f Mhash/s\n", (n_attempts / elapsed) / 1e6);
            break;
        }
        
    }

    return 0;
}
