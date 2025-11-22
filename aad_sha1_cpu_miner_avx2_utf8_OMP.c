#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"
#include <immintrin.h>
#include <omp.h>

#define N_LANES 8   // AVX2 = 256 bits / 32 bits por int = 8 lanes

static inline int prefix_matches_aad2025(u32_t first_word) {
    return first_word == 0xAAD20250u;
}


// LUT: tabela de próximo valor alfanumérico (muito mais rápido que if-else)
static const u08_t NEXT_ALNUM[256] = {
    ['0']=  '1', ['1']=  '2', ['2']=  '3', ['3']=  '4', ['4']=  '5',
    ['5']=  '6', ['6']=  '7', ['7']=  '8', ['8']=  '9', ['9']=  'a',
    ['a']=  'b', ['b']=  'c', ['c']=  'd', ['d']=  'e', ['e']=  'f',
    ['f']=  'g', ['g']=  'h', ['h']=  'i', ['i']=  'j', ['j']=  'k',
    ['k']=  'l', ['l']=  'm', ['m']=  'n', ['n']=  'o', ['o']=  'p',
    ['p']=  'q', ['q']=  'r', ['r']=  's', ['s']=  't', ['t']=  'u',
    ['u']=  'v', ['v']=  'w', ['w']=  'x', ['x']=  'y', ['y']=  'z',
    ['z']=  'A', ['A']=  'B', ['B']=  'C', ['C']=  'D', ['D']=  'E',
    ['E']=  'F', ['F']=  'G', ['G']=  'H', ['H']=  'I', ['I']=  'J',
    ['J']=  'K', ['K']=  'L', ['L']=  'M', ['M']=  'N', ['N']=  'O',
    ['O']=  'P', ['P']=  'Q', ['Q']=  'R', ['R']=  'S', ['S']=  'T',
    ['T']=  'U', ['U']=  'V', ['V']=  'W', ['W']=  'X', ['X']=  'Y',
    ['Y']=  'Z', ['Z']=  '0',
};

// LUT para caracteres alfanuméricos (para init aleatória)
static const u08_t ALPHANUMERIC[62] = {
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
};

// Gera um byte alfanumérico aleatório (muito mais rápido com LUT)
static inline u08_t random_alphanumeric(void) {
    return ALPHANUMERIC[rand() % 62];
}

static inline u08_t next_alphanumeric(u08_t current) {
    return NEXT_ALNUM[current];
}

int main(void)
{
    unsigned long long total_attempts = 0ULL;
    const unsigned long long REPORT_INTERVAL = 100000000ULL;
    const char template[12] = "DETI coin 2 ";
    double global_start_time = omp_get_wtime();
    double last_report_time = global_start_time;
    srand((unsigned int)time(NULL));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        unsigned long long n_attempts = 0ULL;
        u08_t msg[N_LANES * 56] __attribute__((aligned(32)));
        v8si coin[14];   // AVX2: 8 lanes de 32 bits
        v8si hash[5];

        // Inicialização aleatória do nonce
        for(int lane=0; lane<N_LANES; lane++) {
            // copia o template diretamente
            for(int i=0;i<12;i++)
                msg[lane*56 + i] = (u08_t)template[i];

            // padding
            msg[lane*56 + 54] = '\n';
            msg[lane*56 + 55] = 0x80;

            // nonce inicial aleatório nos 42 bytes
            for(int j=0; j<42; j++) {
                u08_t b;
                do {
                    b = (u08_t)(random_alphanumeric());   // 0x00..0x7F
                    if (b == 0x0A) b = 0x0B;      // evitar newline
                } while (b >= 0x80);
                msg[lane*56 + 12 + j] = b;
            }
        }

        u32_t coin_lane[14];

        while(1) {
            // incrementa nonce de 42 bytes por lane (little-endian)
            for (int lane=0; lane<N_LANES; lane++) {
                u08_t *ptr = &msg[lane*56 + 12];
                for(int j=0; j<42; j++) {
                    u08_t *p = &ptr[j];
                    *p = next_alphanumeric(*p);
                    if (*p != '0') break;  // carry propagation ('0' indica wrap)
                }
            }
            n_attempts += N_LANES;

            // monta coin_u32 a partir de msg com byte-swap
            u32_t *coin_u32 = (u32_t*)coin;
            u32_t *msg_u32  = (u32_t*)msg;
            for (int i = 0; i < 14; ++i) {
                for (int lane = 0; lane < N_LANES; ++lane) {
                    uint32_t w = msg_u32[lane * 14 + i];            
                    coin_u32[i * N_LANES + lane] = __builtin_bswap32(w);
                }
            }

            // calcula SHA1 AVX2
            sha1_avx2((v8si*)coin, (v8si*)hash);

            u32_t *hash_u32 = (u32_t*)hash;
            for(int lane=0; lane<N_LANES; lane++) {
                u32_t h0_of_lane = hash_u32[0 * N_LANES + lane];

                if(prefix_matches_aad2025(h0_of_lane)) {
                    u08_t *base = &msg[lane*56];
                    printf("Nonce logical (lane %d): ", lane);
                    for (int j = 0; j < 42; ++j) {
                        int idx = ((12 + j) ^ 3);
                        printf("%02X ", base[idx]);
                    }
                    printf("\n");

                    // Guarda coin de forma consistente
                    for (int i = 0; i < 14; ++i) {
                        uint32_t w_msg = msg_u32[lane * 14 + i];
                        coin_lane[i] = __builtin_bswap32(w_msg);
                        printf("Coin lane %d, word %d: %08X, char: %s\n", lane, i, coin_lane[i], (char*)&coin_lane[i]); // DEBUG
                    }

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
