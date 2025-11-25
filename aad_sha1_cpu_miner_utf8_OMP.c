#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"
#include <immintrin.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    double elapsed = 0.0;
    double time_limit = 0.0; // 0 = sem limite
    const char *name = NULL;
    unsigned long long thread_attempts[128] = {0}; // suporte até 128 threads


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

    //const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
    //int charset_len = sizeof(charset) - 1;

    // array de índices para cada posição da moeda variável (12..53)
    unsigned int indices[42] = {0};  // já inicializado em 0

    printf("Starting CPU miner (without SIMD)...\n");


    #define MIN_CHAR 0x20
    #define MAX_CHAR 0x9F
    #define NONCE_BYTES (42 - name_len)

    unsigned long long n_attempts_total = 0ULL;
    double start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        unsigned long long n_attempts = 0ULL;
        int tid = omp_get_thread_num();
        int myid = tid;  // alias simples

        while(1) {


        // gerar nova moeda (incrementar nonce)
        // soma 1 ao byte 12, se overflow, soma 1 ao byte 13, e assim sucessivamente
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
        n_attempts += 1;

        // verificar se a hash é válida, u -> unsigned -> não negativo
        if (hash[0] == 0xAAD20250u)
        {
            #pragma omp critical
            {
                printf("\033[1;32m Found DETI coin after %llu attempts!\033[0m\n", n_attempts);
                save_coin(coin);
	            save_coin(NULL);    
            }
        } 
        
            double now = omp_get_wtime();
            double elapsed_local = now - start_time;

            if (time_limit > 0.0 && elapsed_local >= time_limit) {
                /* guarda contagens locais numa slot partilhada para agregação */
                thread_attempts[myid] += n_attempts;
                n_attempts = 0ULL;
                        
                /* sincronizar todas as threads: damos tempo para as outras escreverem a sua contagem */
                #pragma omp barrier
                        
                /* apenas a master agrega e imprime */
                #pragma omp master
                {
                    n_attempts_total = 0ULL;
                    int nt = omp_get_num_threads();
                    for (int t = 0; t < nt; t++)
                        n_attempts_total += thread_attempts[t];
                
                    double final_elapsed = omp_get_wtime() - start_time;
                    printf("=== Miner stopped ===\n");
                    printf("Total attempts: %llu\n", n_attempts_total);
                    printf("Elapsed time: %.2f seconds\n", final_elapsed);
                    printf("Average rate: %.2f Mhashes/s\n", (double)n_attempts_total / final_elapsed / 1e6);
                    printf("=== Tempo limite atingido ===\n");
                }
            
                /* garante que a master já fez a impressão antes de sair */
                #pragma omp barrier
            
                /* pede a paragem a todas as threads */
                //atomic_store(&stop_flag, 1);
            
                /* sai do laço (cada thread) */
                break;
            }
        }
    }

    return 0;
}
