#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"
#include <immintrin.h>

#define N_LANES 8   // AVX2 = 256 bits / 32 bits por int = 8 lanes

static inline int prefix_matches_aad2025(u32_t first_word) {
    return first_word == 0xAAD20250u;
}

// Valida se o nonce contém apenas bytes válidos (sem 0x0A e sem bytes >= 0x80)


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
    unsigned long long n_attempts = 0ULL;
    double elapsed;

    u08_t msg[N_LANES * 56] __attribute__((aligned(32)));
    v8si coin[14];   // AVX2: 8 lanes de 32 bits
    v8si hash[5];

    const char template[12] = "DETI coin 2 ";

    // Inicialização aleatória do nonce
    srand((unsigned int)time(NULL));

    for(int lane=0; lane<N_LANES; lane++) {
        // copia o template
        for(int i=0;i<12;i++)
            msg[lane*56 + (i^3)] = (u08_t)template[i];

        // padding
        msg[lane*56 + (54^3)] = '\n';
        msg[lane*56 + (55^3)] = 0x80;

        // nonce inicial aleatório nos 42 bytes
        for(int j=0; j<42; j++) {
            u08_t b;
            do {
                b = (u08_t)(random_alphanumeric());   // 0x00..0x7F
            } while (b >= 0x80);              // redundante, por segurança
            msg[lane*56 + ((12 + j) ^ 3)] = b;
        }
    }

    u32_t coin_lane[14];
    time_measurement();

    while(1) {
        // incrementa nonce de 42 bytes por lane (como little-endian)
        // com validação UTF-8: salta 0x0A e bytes >= 0x80
                // incrementa nonce de 42 bytes por lane (como little-endian)
        // com validação UTF-8: salta 0x0A e bytes >= 0x80
        for (int lane = 0; lane < N_LANES; lane++) {
            for (int j = 0; j < 42; j++) {
                int idx = ((12 + j) ^ 3);               // usa o mesmo mapeamento que no init
                u08_t *p = &msg[lane*56 + idx];

                // incrementa com carry
                (*p)= next_alphanumeric(*p);
                if (*p != '0')  // carry propagation ('0' indica wrap)
                    break;
            }
        }


        // monta coin_u32 a partir de msg com byte-swap
        u32_t *coin_u32 = (u32_t*)coin;
        u32_t *msg_u32  = (u32_t*)msg;
        // monta coin_u32 a partir de msg
        for(int i=0;i<14;i++) {
            for(int lane=0; lane<N_LANES; lane++) {
                // copiar os bytes diretamente do msg
                coin_u32[i*N_LANES + lane] = msg_u32[lane*14 + i]; // sem bswap aqui
            }
        }
        
        

        // calcula SHA1 AVX2
        sha1_avx2((v8si*)coin, (v8si*)hash);

        u32_t *hash_u32 = (u32_t*)hash;
        for(int lane=0; lane<N_LANES; lane++) {
            u32_t h0_of_lane = hash_u32[0 * N_LANES + lane];
            if(h0_of_lane == 0xEB90509u) {
                printf("Found target hash in lane %d: %08X\n", lane, h0_of_lane);
            }
            if(prefix_matches_aad2025(h0_of_lane)) {
                // Valida que o nonce NÃO contém 0x0A ou bytes >= 0x80

                // exemplo de impressão do nonce lógico (42 bytes)
                u08_t *base = &msg[lane*56];
                printf("Nonce logical (lane %d): ", lane);
                for (int j = 0; j < 42; ++j) {
                    int idx = ((12 + j) ^ 3);
                    printf("%02X ", base[idx]);
                }
                printf("\n");
                
                for(int i=0;i<14;i++)
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
            fflush(stdout);
            if (elapsed > 0.0)
                printf("%.3f Mhashes/s\n", (double)n_attempts / (elapsed * 1e6));
        }
    }

    return 0;
}
