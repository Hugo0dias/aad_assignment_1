#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../aad_data_types.h"
#include "../aad_sha1_cpu.h"
#include "../aad_utilities.h"
#include "../aad_vault.h"

/*
emcc aad_sha1_cpu_miner.c -O3 -s WASM=1 -s EXPORTED_FUNCTIONS='["_main"]' -s MODULARIZE=1 -s EXPORT_NAME='createModule' -o deti_coin.js
*/

int main() {
    double time_limit = 0.0;
    const char *name = NULL;

    u32_t coin[14];
    u32_t hash[5];
    u08_t *msg = (u08_t *)coin;
    unsigned long long n_attempts = 0ULL;

    memset(msg, 0, 56);

    const char template[12] = "DETI coin 2 ";
    for (int i = 0; i < 12; i++)
        msg[i ^ 3] = (u08_t)template[i];
    msg[54 ^ 3] = '\n';
    msg[55 ^ 3] = 0x80;

    printf("Starting CPU miner (without SIMD)...\n");

    double elapsed = 0.0;
    time_measurement();

    #define MIN_CHAR 0x20
    #define MAX_CHAR 0x9F
    unsigned int indices[42] = {0};

    while(1) {
        for (int i = 0; i < 42; i++)
            msg[(12 + i) ^ 3] = (u08_t)(MIN_CHAR + indices[i]);

        for (int i = 0; i < 42; i++) {
            indices[i]++;
            if (indices[i] > (MAX_CHAR - MIN_CHAR))
                indices[i] = 0;
            else
                break;
        }

        if (n_attempts < 10) {
			// msg = coin apontam para mesma memoria
            printf("Tentativa %llu:\n", n_attempts + 1);

            printf("Coin bytes (as seen by save_coin() / SHA1 CPU):\n");
			for (int i = 0; i < 56; i++) {
				printf("%02x ", msg[i]);
				if ((i + 1) % 16 == 0) printf("\n");
			}
			printf("\n");
        }

        sha1(coin, hash);
        n_attempts++;

        if (hash[0] == 0xAAD20250u) {
            printf("Found DETI coin after %llu attempts!\n", n_attempts);
            save_coin(coin);
            save_coin(NULL);
        }

        time_measurement();
        elapsed += wall_time_delta();
        if (time_limit > 0.0 && elapsed >= time_limit) break;
    }
}


