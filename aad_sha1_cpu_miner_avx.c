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

static void print_bytes(const char *title, u08_t *p, int n) {
    printf("%s (%d bytes):\n", title, n);
    for (int i = 0; i < n; i++) {
        printf("%02X ", p[i]);
        if ((i+1)%16 == 0) printf("\n");
    }
    if (n % 16) printf("\n");
}

static void print_v4si_words(const char *title, v4si *v, int n_words) {
    printf("%s:\n", title);
    for (int w = 0; w < n_words; w++) {
        printf("W%02d: ", w);
        for (int lane = 0; lane < N_LANES; lane++)
            printf("%08X ", v[w][lane]);
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    #define BLOCK_SIZE 5

    unsigned long long base_nonce[N_LANES] = {0};  // offset base para cada lane
    unsigned long long lane_nonce[N_LANES] = {0};  // nonce relativo à lane

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

    u08_t msg[N_LANES * 56];
    v4si coin[14];
    v4si hash[5];
    memset(msg, 0, sizeof(msg));

    // reserva 1os 12 bytes para o cabeçalho fixo da moeda, para cada uma das lanes
    // Monta msg por lane
    const char template[12] = "DETI coin 2 ";

    // Calcula tamanho do nome
    size_t name_len = 0;
    if (name != NULL) {
        name_len = strlen(name);
        if (name_len > 42) name_len = 42;
    }

    for (int lane = 0; lane < N_LANES; lane++) {
        // 1. Prefixo
        for (int i = 0; i < 12; i++)
            msg[lane*56 + i] = (u08_t)template[i];

        // 2. Nome (se houver)
        for (size_t i = 0; i < name_len; i++)
            msg[lane*56 + 12 + i] = (u08_t)name[i];

        // 4. Terminadores -> vao ser sobrescritos pelo nonce maybe
        msg[lane*56 + 54] = '\n';
        msg[lane*56 + 55] = 0x80;
    }

    // ponteiros para nonces (bytes 12..53)
    u08_t *nonce[N_LANES];
    for (int lane = 0; lane < N_LANES; lane++)
        nonce[lane] = &msg[lane*56 + 12 + name_len];

    // coin_lane = 1 coin temp
    u32_t coin_lane[14];
    
    double elapsed = 0.0;
    time_measurement();

    // inicializa offsets diferentes para cada lane
    for (int lane = 0; lane < N_LANES; lane++) {
        base_nonce[lane] = (unsigned long long)lane * (unsigned long long)BLOCK_SIZE;
        lane_nonce[lane] = 0;
    }

    // opcional: limpa os bytes do nonce no buffer msg (para ter estado determinístico)
    for (int lane = 0; lane < N_LANES; lane++) {
        for (int i = 0; i < 42; i++)
            nonce[lane][i] = 0;
    }

    while(1) {

        for (int lane = 0; lane < N_LANES; lane++) {
            lane_nonce[lane]++;      // incrementa nonce relativo
            if (lane_nonce[lane] >= BLOCK_SIZE) {
                lane_nonce[lane] = 0;
                base_nonce[lane] += (unsigned long long)BLOCK_SIZE * (unsigned long long)N_LANES; // avança bloco
            }

            unsigned long long final_nonce = base_nonce[lane] + lane_nonce[lane];

            //printf("%llu", final_nonce);
            //printf("\n");

            // escreve final_nonce em little-endian nos 42 bytes do nonce,
            // substitui 0x0A por 0x08 (backspace) se precisares
            // limpa todos os 42 bytes do nonce
            u08_t vbytes[42-name_len];
            unsigned long long t = final_nonce;
            for (int i = 0; i < 42-name_len; i++) {
                vbytes[i] = (u08_t)(t & 0xFFu);
                t >>= 8;
            }
            for (int i = 0; i < 42 -name_len; i++)
                if (vbytes[i] == 0x0A) vbytes[i] = 0x08;
            for (int i = 0; i < 42 -name_len; i++)
                msg[lane*56 + 12 + name_len + i] = vbytes[i];
            
            // sobrescrever o q foi sobrescrito, ahahahahah
            msg[lane*56 + 54] = '\n';
            msg[lane*56 + 55] = 0x80;
        }

        /*for (int lane = 0; lane < N_LANES; lane++) { unsigned long long trial = n_attempts + lane + 1; 
            printf("Tentativa %llu, Lane %d, nonce = ", trial, lane); 
            for (int i = 0; i < 42; i++) 
                printf("%02X", nonce[lane][i]); 
            printf("\n"); }
        
        if (n_attempts < 4000) { 
            printf("\nTentativa %llu:\n", n_attempts + 1); 
            for (int lane = 0; lane < N_LANES; lane++) { 
                printf(" Lane %d:\n", lane); 
                for (int i = 0; i < 56; i++) { 
                    printf("%02X ", msg[lane*56 + (i^3)]); 
                    if ((i + 1) % 16 == 0) printf("\n"); 
                } printf("\n"); 
            } 
            printf("\n"); 
        }*/
        
        // bem
        for(int i = 0; i < 14; i++) {       // cada word
            for(int lane = 0; lane < N_LANES; lane++) {
                u32_t word = 0;
                for(int b = 0; b < 4; b++) {
                    word |= ((u32_t)msg[lane*56 + i*4 + b]) << (8*b);  // pega word i de cada lane
                }
                coin[i][lane] = word;  // word i de cada lane
            }
        }
        //print_v4si_words("COIN (real, LE)", coin, 14);
        
        v4si coin_swap[14];
        for (int w = 0; w < 14; w++)
        for (int lane = 0; lane < N_LANES; lane++)
            coin_swap[w][lane] = __builtin_bswap32(coin[w][lane]);

        //print_v4si_words("COIN_SWAP (^3, BE)", coin_swap, 14);


        // chama a versão AVX da sha1 que produz 4 hashes em paralelo
        sha1_avx((v4si*)coin_swap, (v4si*)hash);

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
                // extrair a coin dessa lane (reconstruir 14 words)
                for(int i=0;i<14;i++)
                    coin_lane[i] = coin[i][lane];

                save_coin(coin_lane);
                u08_t *coin_bytes = (u08_t *)coin_lane;
                printf("\033[1;32mFound DETI coin after %llu attempts!\033[0m\n", n_attempts);

                printf("Coin bytes (human order):\n");
                for (int i = 0; i < 56; i++) {
                    printf("%02X ", coin_bytes[i]);
                    if ((i + 1) % 16 == 0) printf("\n");
                }
                printf("\nAs string (printable ASCII, \\x?? for non-printables):\n");
                for (int i = 0; i < 56; i++) {
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


