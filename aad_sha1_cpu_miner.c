#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"

int main(void)
{
    u32_t coin[14];    // 14 * 4bytes = 56 bytes -> tamanho da moeda
    u32_t hash[5];     // 5  * 4bytes = 20 bytes -> tamanho hash sha1
    /*
    coin -> 14 caixas grandes com 4 moedas cada
    msg  -> ignora a caixas grandes e pega moeda a moeda (byte a byte) -> 8bits = 1byte
    */
    u08_t *msg = (u08_t *)coin;

    //double elapsed;
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

    printf("Starting CPU miner (without SIMD)...\n");

    double total_time = 0.0;
    unsigned long long total_hashes = 0ULL;

    time_measurement(); // inicializa

    while(1) {
        // gerar nova moeda (incrementar nonce)
        // soma 1 ao byte 12, se overflow, soma 1 ao byte 13, e assim sucessivamente
        for (int i = 12; i < 54; i++) {
            if (++msg[i] != 0) break;
        }

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
        
        if (n_attempts % 1000000ULL == 0ULL) {
            time_measurement();
            double block_time = wall_time_delta(); // tempo do bloco
            total_time += block_time;
            total_hashes += 1000000ULL;

            double avg_rate = (total_hashes / total_time) / 1e6;
            printf("Último bloco: %.3f s (%.2f Mhash/s) | Média: %.2f Mhash/s\n",
                block_time, 1.0 / block_time, avg_rate);
        }
    }

    return 0;
}
