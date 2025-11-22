#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_utilities.h"
#include "aad_vault.h"
#include <immintrin.h>
#include <omp.h>

#define N_LANES 4
#define COIN_SIZE 56 

static inline int prefix_matches_aad2025(u32_t first_word) {
    return first_word == 0xAAD20250u; // cheque exacto do word H0
}

int main(int argc, char *argv[])
{
    double time_limit = 0.0;
    const char *name = NULL;
    unsigned long long thread_attempts[128] = {0}; // suporte até 128 threads

    if (argc > 1) {
        time_limit = atof(argv[1]);
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

    size_t name_len = name ? strlen(name) : 0;
    if (name_len > 42) name_len = 42;

    const char template[12] = "DETI coin 2 ";
    unsigned long long n_attempts_total = 0ULL;
    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned long long n_attempts = 0ULL;
        int myid = tid;  // alias simples


        u08_t msg[N_LANES * COIN_SIZE];
        v4si coin[14];
        v4si hash[5];
        u08_t nonce[N_LANES][42];
        u08_t nonce2[N_LANES][42];

        // Inicializa lanes com escalonamento PERMANENTE por thread E lane
        // Cada combinação (thread, lane) tem byte 0 ÚNICO
        // Fórmula: 0x20 + (tid * N_LANES + lane) * 4
        // Garante que cada (thread, lane) tem um valor único
        
        for(int lane=0; lane<N_LANES; lane++){
            memcpy(&msg[lane*COIN_SIZE], template, 12);
            if(name_len) memcpy(&msg[lane*COIN_SIZE+12], name, name_len);

            int avail = 42 - (int)name_len;
            // Escalonamento ÚNICO por (thread, lane) - sem colisões
            int unique_id = tid * N_LANES + lane;  // ID único para cada (thread, lane)
            nonce[lane][0] = 0x20 + (unique_id % 128);  // Distribui em 0x20-0x9F
            
            for(int i=1; i<avail; i++)
                nonce[lane][i] = 0x20; // restante inicializado

            // Copia nonce para msg
            memcpy(&msg[lane*COIN_SIZE+12+name_len], &nonce[lane][0], avail);

            msg[lane*COIN_SIZE+54] = '\n';
            msg[lane*COIN_SIZE+55] = 0x80;
        }

        const int MIN_CHAR = 0x20;
        const int MAX_CHAR = 0x9F;
        int avail = 42 - (int)name_len;
        int base_idx = 0;                  // Byte 0 é o índice base FIXO
        int odometer_start = 1;            // Odômetro começa no byte 1
        int odometer_end = avail - 1;      // Termina no último byte

        // Define nonce pré-definido para comparação
        u08_t target_nonce[42] = {
            0x3D, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x45, 0x3E
        };

        while(1) {
            // Incrementa odômetro alfanumérico por lane (bytes 1 até avail-1)
            // Byte 0 NUNCA é incrementado para manter diferenciação entre lanes
            for(int lane=0; lane<N_LANES; lane++){
                int carry=1;
                
                if (n_attempts_total >= 5000000000){
                    #pragma omp critical
                    {
                        printf("Entered critical section for nonce check\n");
                        // Copia nonce para verificação
                        //memcpy(nonce2[lane], nonce[lane], avail);
                        //// 
                        //// // Compara com nonce pré-definido
                        //if(memcmp(nonce2[lane], target_nonce, avail) == 0){
                        //    printf("\n*** Thread %d Lane %d ENCONTROU NONCE PRÉ-DEFINIDO! ***\n", tid, lane);
                        //    printf("Nonce: ");
                        //    for(int i=0; i<avail; i++){
                        //        printf("%02X", nonce2[lane][i]);
                        //    }
                        //    printf("\n\n");
                        
                        printf("Thread %d Lane %d: ", tid, lane);
                        for(int i=0; i<avail; i++){
                            printf("%02X ", nonce[lane][i]);
                        }
                        printf("\n");
                        //}
                    }
                }
                
                for(int i=odometer_end; i>=odometer_start && carry; i--){
                    nonce[lane][i]++;
                    if(nonce[lane][i] > MAX_CHAR){
                        nonce[lane][i] = MIN_CHAR;
                        carry = 1;
                    } else carry = 0;
                    msg[lane*COIN_SIZE + 12 + i + name_len] = nonce[lane][i];
                }
                // Se odômetro fez wrap completo, poderia incrementar byte 0
                // MAS NÃO FAZEMOS para manter diferenciação entre lanes!
            }

            n_attempts += N_LANES;

            // Monta coin em big-endian
            for(int w=0; w<14; w++){
                for(int lane=0; lane<N_LANES; lane++){
                    u32_t word = 0;
                    for(int b=0;b<4;b++){
                        int idx=w*4 + b;
                        if(idx < COIN_SIZE)
                            word |= ((u32_t)msg[lane*COIN_SIZE + (idx^3)]) << (8*b);
                    }
                    coin[w][lane] = word;
                }
            }

            // SHA1 AVX para 4 lanes
            sha1_avx(coin, hash);

            u32_t *hash_u32 = (u32_t*)hash;
            for(int lane=0; lane<N_LANES; lane++){
                u32_t h0 = hash_u32[0*N_LANES + lane];

                if(prefix_matches_aad2025(h0)){
                    u32_t coin_lane_arr[14];
                    for(int w=0; w<14; w++)
                        coin_lane_arr[w] = coin[w][lane];

                    #pragma omp critical
                    {
                        save_coin(coin_lane_arr);
                        save_coin(NULL);
                        printf("\033[1;32mThread %d found DETI coin after %llu attempts!\033[0m\n",
                               tid, n_attempts_total + n_attempts);
                    }
                }
            }

            // Reporting e checagem de tempo
            if(n_attempts % 100000000 == 0){
                thread_attempts[myid] += n_attempts;
                n_attempts = 0ULL;

                #pragma omp master
                {
                    n_attempts_total = 0ULL;
                    int nt = omp_get_num_threads();
                    for (int t = 0; t < nt; t++)
                        n_attempts_total += thread_attempts[t];
                }


                #pragma omp master
                {
                    double now = omp_get_wtime();
                    double elapsed = now - start_time;
                    printf("Total attempts: %llu, Elapsed: %.2f sec\n", n_attempts_total, elapsed);

                    if(time_limit > 0.0 && elapsed >= time_limit){
                        printf("=== Miner stopped ===\n");
                        printf("Total attempts: %llu\n", n_attempts_total);
                        double elapsed = omp_get_wtime() - start_time;
                        printf("Average rate: %.2f Mhashes/s\n",
                                  (double)n_attempts_total / elapsed / 1e6);
                        printf("=== Tempo limite atingido ===\n");
                        exit(0);
                    }
                }
            }
        }
    }



    return 0;
}
