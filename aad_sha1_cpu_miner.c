#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"

int main(int argc, char *argv[])
{
    double time_limit = 0.0; // 0 = sem limite
    const char *name = NULL;

    if (argc > 1) {
        time_limit = atof(argv[1]); // converte string -> double
        if (time_limit <= 0.0) {
            printf("Invalid time value. Usage: %s [time_in_seconds] [name]\n", argv[0]);
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


    u32_t coin[14];    // 14 * 4bytes = 56 bytes -> tamanho da moeda
    u32_t hash[5];     // 5  * 4bytes = 20 bytes -> tamanho hash sha1
    /*
    coin -> 14 caixas grandes com 4 moedas cada
    msg  -> ignora a caixas grandes e pega moeda a moeda (byte a byte) -> 8bits = 1byte
    */
    u08_t *msg = (u08_t *)coin;

    unsigned long long n_attempts = 0ULL;

    memset(msg, 0, 56);    // vai ao end. memoria msg e mete a 0s 56 bytes (limpa lixo)

    /*
    strcpy num array de u08_t é alinhado como u32_t, o SHA1 CPU espera os bytes organizados com ^3 (endianness)
    strcpy((char *)msg, "DETI coin 2 ");  // não respeita a troca de endian
    Por isso o save_coin() vê “DIET” em vez de “DETI”
    */

    // reserva 1os 12 bytes para o cabeçalho fixo da moeda
    const char template[12] = "DETI coin 2 ";    
    for (int i = 0; i < 12; i++)
        msg[i ^ 3] = (u08_t)template[i];    // aplica XOR 3 em cada byte
    msg[54 ^ 3] = '\n';   // penúltimo byte é \n
    msg[55 ^ 3] = 0x80;    // último byte é padding

    // se houver nome, fixa na posição 12
    size_t name_len = 0;
    if (name != NULL) {
        name_len = strlen(name);
        if (name_len > 42) name_len = 42; // verifica se cabe nos bytes 12..53
        for (size_t i = 0; i < name_len; i++)
            msg[(12 + i) ^ 3] = (u08_t)name[i];
    }

    printf("Starting CPU miner (without SIMD)...\n");

    double elapsed = 0.0;
    time_measurement(); // inicializa

    while(1) {
        // gerar nova moeda (incrementar nonce)
        // soma 1 ao byte 12, se overflow, soma 1 ao byte 13, e assim sucessivamente
        for (int i = 12 + name_len; i < 54; i++) {
            if (++msg[i^3] == '\n')  // se for 0x0A, incrementa again
                ++msg[i^3];
            if (msg[i^3] != 0) break;
        }

        /*if (n_attempts < 1000) {
			// msg = coin apontam para mesma memoria
            printf("Tentativa %llu:\n", n_attempts + 1);

            printf("Coin bytes (as seen by save_coin() / SHA1 CPU):\n");
			for (int i = 0; i < 56; i++) {
				printf("%02x ", msg[i]);
				if ((i + 1) % 16 == 0) printf("\n");
			}
			printf("\n");
        }*/

        // calcular hash sha1
        sha1(coin, hash);
        n_attempts++;

        // verificar se a hash é válida, u -> unsigned -> não negativo
        if (hash[0] == 0xAAD20250u)
        {
            printf("\033[1;32m Found DETI coin after %llu attempts!\033[0m\n", n_attempts);
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

    return 0;
}
