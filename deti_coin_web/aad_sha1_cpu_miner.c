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
    double time_limit = 60;
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

    size_t name_len = 0;
    if (name != NULL) {
        name_len = strlen(name);
        if (name_len > 42) name_len = 42; // verifica se cabe nos bytes 12..53
        for (size_t i = 0; i < name_len; i++)
            msg[(12 + i) ^ 3] = (u08_t)name[i];
    }

    unsigned int indices[42] = {0};

    printf("Starting CPU miner (without SIMD)...\n");

    double elapsed = 0.0;
    time_measurement(); // inicializa

    #define MIN_CHAR 0x20
    #define MAX_CHAR 0x9F
    #define NONCE_BYTES (42 - name_len)

    while(1) {
        for (int i = 0; i < NONCE_BYTES; i++) {
            msg[(12 + name_len + i) ^ 3] = (u08_t)(MIN_CHAR + indices[i]);
        }

        // incrementa odômetro
        for (int i = 0; i < NONCE_BYTES; i++) {
            indices[i]++;
            if (indices[i] > (MAX_CHAR - MIN_CHAR))
                indices[i] = 0;  // carry para o próximo byte
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

        time_measurement();                 // mede nova amostra
        elapsed += wall_time_delta();       // acumula tempo decorrido

        if (time_limit > 0.0 && elapsed >= time_limit) {
            printf("\n=== Tempo limite atingido (%.2f / %.2f) ===\n", elapsed, time_limit);
            printf("Tentativas totais: %llu\n", n_attempts);
            printf("Taxa média: %.2f Mhash/s\n", (n_attempts / elapsed) / 1e6);
            break;
        }
    }
}


