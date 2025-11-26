
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"
#include <immintrin.h>

#define N_LANES 16
#define COIN_SIZE 56 

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

    unsigned long long n_attempts = 0ULL;

    u08_t msg[N_LANES * COIN_SIZE] __attribute__((aligned(64)));
    v16si coin[14];
    v16si hash[5];
    u08_t coin_lane[COIN_SIZE];
    memset(msg, 0, sizeof(msg));


    // reserva 1os 12 bytes para o cabeçalho fixo da moeda, para cada uma das lanes
    // Monta msg por lane
    const char template[12] = "DETI coin 2 ";
    size_t name_len = name ? strlen(name) : 0;
    if(name_len > 42) name_len = 42;

    for (int lane = 0; lane < N_LANES; lane++) {
        for (int i = 0; i < 12; i++)
            msg[lane*COIN_SIZE + i] = template[i];
        for (size_t i = 0; i < name_len; i++)
            msg[lane*COIN_SIZE + 12 + i] = name[i];
        for (int i = 0; i < 42 - name_len; i++)
            msg[lane*COIN_SIZE + 12 + name_len + i] = 0;
        msg[lane*COIN_SIZE + 54] = '\n';   // byte 54
        msg[lane*COIN_SIZE + 55] = 0x80;   // byte 55
    }

    int avail = 42 - (int)name_len;
    if (avail < 0) avail = 0;

    /* se não houver espaço para nonce, aborta (ou trata à tua maneira) */
    if (avail == 0) {
        fprintf(stderr, "Erro: name_len=%zu usa todos os 42 bytes de nonce. Nada para iterar.\n", name_len);
        return 1;
    }

    /* configuração de intervalo (configurável) */
    #define NONCE_LEN avail
    #define MIN_CHAR 0x20
    #define MAX_CHAR 0x7E

    unsigned long long base_nonce[N_LANES] = {0};
    unsigned long long lane_nonce[N_LANES] = {0};
    const int total_values = MAX_CHAR - MIN_CHAR + 1;
    const int values_per_lane = total_values / N_LANES;

    u08_t nonce[N_LANES][NONCE_LEN]; // buffer do nonce por lane

    // inicializa nonce
    for (int lane = 0; lane < N_LANES; lane++) {
        for (int i = 0; i < NONCE_LEN; i++)
            nonce[lane][i] = MIN_CHAR + lane * values_per_lane; // cada lane começa na sua fatia
    }


    
    double elapsed = 0.0;
    time_measurement();

    while(1) {

        for (int lane = 0; lane < N_LANES; lane++) {
            unsigned int carry = 1;
            for (int i = NONCE_LEN-1; i >= 0 && carry; i--) {
                nonce[lane][i]++;
                if (nonce[lane][i] > MAX_CHAR) {  // vai até o MAX_CHAR global
                    nonce[lane][i] = MIN_CHAR;    // reinicia
                    carry = 1;
                } else {
                    carry = 0;
                }
            }
            memcpy(&msg[lane*COIN_SIZE + 12 + name_len], nonce[lane], NONCE_LEN);

            msg[lane*COIN_SIZE + 54] = '\n';
            msg[lane*COIN_SIZE + 55] = 0x80;
        }



        /*for (int lane = 0; lane < N_LANES; lane++) { unsigned long long trial = n_attempts + lane + 1; 
            printf("Tentativa %llu, Lane %d, nonce = ", trial, lane); 
            for (int i = 0; i < 42; i++) 
                printf("%02X", nonce[lane][i]); 
            printf("\n"); 
        }

    
        if (n_attempts < 4000) { 
            printf("\nTentativa %llu:\n", n_attempts + 1); 
            for (int lane = 0; lane < N_LANES; lane++) { 
                printf(" Lane %d:\n", lane); 
                for (int i = 0; i < COIN_SIZE; i++) { 
                    printf("%02X ", msg[lane*COIN_SIZE + (i^3)]); 
                    if ((i + 1) % 16 == 0) printf("\n"); 
                } printf("\n"); 
            } 
            printf("\n"); 
        }*/
        

        for(int w=0; w<14; w++) {
            for(int lane=0; lane<N_LANES; lane++) {
                u32_t word = 0;
                for(int b=0; b<4; b++) {
                    int idx = w*4 + b;
                    if(idx < COIN_SIZE)
                        word |= ((u32_t)msg[lane*COIN_SIZE + (idx ^ 3)]) << (8*b);
                }
                coin[w][lane] = word;
            }
        }


        // chama a versão AVX da sha1 que produz 4 hashes em paralelo
        sha1_avx512f((v16si*)coin, (v16si*)hash);

        //print_v4si_words("SHA1 HASH (interleaved lanes)", hash, 5);

        // layout: ((u32_t*)hash) = [h0_l0,h0_l1,h0_l2,h0_l3, h1_l0,h1_l1,...]
        u32_t *hash_u32 = (u32_t*)hash;
        /*printf("\nHASH per lane (with endian correction):\n");
        for (int lane = 0; lane < N_LANES; lane++) {
            u32_t h0_le = hash_u32[0*N_LANES + lane];
            u32_t h0_be = __builtin_bswap32(h0_le);
            printf("Lane %d: H0 = LE=%08X  BE=%08X\n",
                lane, h0_le, h0_be);
        }*/
        for(int lane=0; lane<N_LANES; lane++) {
            u32_t h0_of_lane = hash_u32[0 * N_LANES + lane]; // índice = word_index*4 + lane
            if(prefix_matches_aad2025(h0_of_lane)) {

                u32_t coin_lane_arr[14];
                for(int w=0; w<14; w++)
                    coin_lane_arr[w] = coin[w][lane];  // pega só a lane válida
                save_coin(coin_lane_arr);


                // converte para bytes para print
                u08_t coin_bytes[COIN_SIZE];
                for(int w=0; w<14; w++)
                    for(int b=0; b<4; b++)
                        coin_bytes[w*4 + b] = (coin_lane_arr[w] >> (8*b)) & 0xFF;

                printf("\033[1;32mFound DETI coin after %llu attempts!\033[0m\n", n_attempts);
                printf("Coin bytes (human order):\n");
                for (int i = 0; i < COIN_SIZE; i++) {
                    printf("%02X ", coin_bytes[i]);
                    if ((i + 1) % 16 == 0) printf("\n");
                }
                printf("\nAs string (printable ASCII, \\x?? for non-printables):\n");
                for (int i = 0; i < COIN_SIZE; i++) {
                    if (coin_bytes[i] >= 32 && coin_bytes[i] <= 126)
                        printf("%c", coin_bytes[i]);
                    else
                        printf("\\x%02X", coin_bytes[i]);
                }
                printf("\n");
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