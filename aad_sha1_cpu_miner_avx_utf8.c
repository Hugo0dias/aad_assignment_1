
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"
#include <immintrin.h>

#define N_LANES 4
#define COIN_SIZE 56 

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

static inline u08_t utf8_next(u08_t b)
{
    b++;

    // evitar control chars e newline
    if (b < 0x20 || b == 0x0A)
        b = 0x20;

    // evitar DEL
    if (b == 0x7F)
        b = 0xC2;  // início UTF-8 extended

    // UTF-8 continuation range opcional (0x80–0xBF)
    // Se queres UTF-8 válido MESMO: permitir isto
    if (b == 0x80)
        return 0x80 + (rand() & 0x3F);

    // lead bytes válidos: 0xC2–0xF4
    if (b == 0xC0 || b == 0xC1)
        b = 0xC2;

    if (b > 0xF4)
        b = 0x20;  // wrap para ASCII

    return b;
}


int main(int argc, char *argv[])
{

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

    u08_t msg[N_LANES * COIN_SIZE];
    v4si coin[14];
    v4si hash[5];
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

    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
    int charset_len = (int)(sizeof(charset) - 1);

    /* para cada lane, um odómetro com avail posições */
    /* para cada lane, um odómetro com avail posições */
    #define BLOCK_SIZE 1 // já tens
    int avail = 42 - (int)name_len;
    unsigned long long lane_block = BLOCK_SIZE; // cada lane processa N nonces por iteração

    unsigned int indices[N_LANES][42];
    memset(indices, 0, sizeof(indices));

    // ponteiros para nonces (bytes 12..53)
    // ponteiros para os bytes 12..53
    u08_t *nonce[N_LANES];
    for (int lane = 0; lane < N_LANES; lane++) {
        nonce[lane] = &msg[lane * COIN_SIZE + 12 + name_len];
    }

    // inicializa cada lane com nonce diferente
    for (int lane = 0; lane < N_LANES; lane++) {
        for (int i = 0; i < avail; i++) {
            nonce[lane][i] = 0x20 + lane;   // valores diferentes entre lanes
        }
    }


    
    double elapsed = 0.0;
    time_measurement();

    /* 1) inicializa base_nonce para escalonar as lanes (evita colisões) */
    u08_t lane_start[N_LANES] = {0x20, 0x40, 0x60, 0x80}; // exemplo
    for(int lane=0; lane<N_LANES; lane++){
        nonce[lane][0] = lane_start[lane];
        for(int i=1; i<avail; i++)
            nonce[lane][i] = 0x20;
    }



    #define MIN_CHAR 0x20   // início do intervalo
    #define MAX_CHAR 0x9F   // fim do intervalo (128 valores = 0x20..0x9F). usa 0x7E se preferires 95

    while(1) {

        // --- parâmetros que podes ajustar ---
        // --------------------------------------------------------------------
        int base_idx = 0; // primeiro byte da lane
        int odometer_start = 1; // após base, ou após name_len
        int odometer_end = avail-1; // último byte disponível do odômetro

        for(int lane=0; lane<N_LANES; lane++){
            int carry = 1;  // sempre começamos incrementando
            for(int i = odometer_end; i >= odometer_start && carry; i--){
                nonce[lane][i] += carry;
                if(nonce[lane][i] > 0x9F){
                    nonce[lane][i] = 0x20; // wrap
                    carry = 1; // carry continua
                } else {
                    carry = 0; // carry resolvido
                }
            }
            // só incrementa base se odômetro completo deu wrap
            if(carry){
                nonce[lane][base_idx]++;
                if(nonce[lane][base_idx] > 0x9F)
                    nonce[lane][base_idx] = 0x20; // wrap da base
            }

            msg[lane*COIN_SIZE + 54] = '\n';   // byte 54
            msg[lane*COIN_SIZE + 55] = 0x80;   // byte 55
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
        sha1_avx((v4si*)coin, (v4si*)hash);

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