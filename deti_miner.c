//
// deti_miner.c
// Arquiteturas de Alto Desempenho 2025/2026
// Host program para mineração de DETI coins com CUDA.
// -> Cada kernel gera e testa mensagens inteiras na GPU (sem transferências de dados massivas)
// -> Apenas o pequeno buffer de resultados é copiado de volta.
//
/*
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_sha1.h"
#include "aad_utilities.h"
#include "aad_cuda_utilities.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

#define THREADS_PER_BLOCK RECOMENDED_CUDA_BLOCK_SIZE
#define COINS_BUFFER_WORDS 1024
#define ITER_PER_THREAD 2048

extern void deti_miner_kernel(u64_t start_nonce, u32_t *coins_storage_area);

//
// Função que lança o kernel e recolhe os resultados
//
static void mine_deti_coins(u64_t start_nonce, int n_blocks)
{
    cuda_data_t cd;
    double kernel_time;
    u32_t *coins_host;

    // configurar contexto CUDA
    cd.device_number = 0;
    cd.cubin_file_name = "deti_miner_kernel.cubin";
    cd.kernel_name = "deti_miner_kernel";
    cd.n_kernel_arguments = 2;

    cd.data_size[0] = COINS_BUFFER_WORDS * sizeof(u32_t); // coins buffer
    cd.data_size[1] = 0;
    initialize_cuda(&cd);

    coins_host = (u32_t *)cd.host_data[0];
    coins_host[0] = 1; // contador de coins
    host_to_device_copy(&cd, 0);

    // definir dimensões de grelha e bloco
    cd.grid_dim_x = n_blocks;
    cd.block_dim_x = THREADS_PER_BLOCK;

    // parâmetros do kernel
    cd.arg[0] = &start_nonce;
    cd.arg[1] = &cd.device_data[0];

    // lançar kernel e medir tempo
    time_measurement();
    lauch_kernel(&cd);
    time_measurement();
    kernel_time = wall_time_delta();

    // copiar resultados de volta
    device_to_host_copy(&cd, 0);

    u32_t used = coins_host[0];
    // 1 coin ocupa = 1 (contador) + (numero * 14), moeda ocupa 14 words
    u32_t n_coins = (used - 1) / 14;
    printf("Kernel run (%.6f s): %u coin(s) found\n", kernel_time, n_coins);

    // guardar coins encontradas
    for (u32_t i = 0; i < n_coins; i++)
        save_coin(&coins_host[1 + i * 14]);
    save_coin(NULL); // flush

    terminate_cuda(&cd);
}

//
// main()
// Executa múltiplos ranges de nonces para explorar o espaço de busca
//
int main(void)
{
    const int n_blocks = 512;
    u64_t start = 0;

    printf("=== DETI Coin CUDA Miner ===\n");

    for (int batch = 0; batch < 100; batch++) {
        mine_deti_coins(start, n_blocks);
        start += (u64_t)n_blocks * THREADS_PER_BLOCK * ITER_PER_THREAD;
    }

    printf("Mineração concluída.\n");
    return 0;
}
*/

// Configurações CUDA escolhidas para um bom desempenho:
// THREADS_PER_BLOCK = 256  -> valor típico e eficiente para GPUs modernas (múltiplos de 32 threads).
// ITER_PER_THREAD  = 2048  -> cada thread faz várias tentativas, reduzindo lançamentos do kernel.
// n_blocks = 512           -> define quantos blocos executam em paralelo na GPU.


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aad_data_types.h"
#include "aad_sha1_cpu.h"
#include "aad_sha1.h"
#include "aad_utilities.h"
#include "aad_cuda_utilities.h"
#include "aad_sha1_cpu.h"
#include "aad_vault.h"

#define THREADS_PER_BLOCK RECOMENDED_CUDA_BLOCK_SIZE
#define COINS_BUFFER_WORDS 1024
#define ITER_PER_THREAD 2048

extern void deti_miner_kernel(u64_t start_nonce, u32_t *coins_storage_area);
//
// Função que lança o kernel e recolhe os resultados
//
static void mine_deti_coins(u64_t start_nonce, int n_blocks, cuda_data_t *cd,
                             const char *name, u08_t name_len)
{
    u32_t *coins_host = (u32_t *)cd->host_data[0];
    coins_host[0] = 1; // contador de coins
    host_to_device_copy(cd, 0);

    // definir dimensões de grelha e bloco
    cd->grid_dim_x = n_blocks;
    cd->block_dim_x = THREADS_PER_BLOCK;

    // alocar memória no device para o nome
    CUdeviceptr name_dev = 0;
    if (name_len > 0) {
        CU_CALL(cuMemAlloc, (&name_dev, name_len));
        CU_CALL(cuMemcpyHtoD, (name_dev, (void*)name, name_len));
    }

    // parâmetros do kernel
    cd->arg[0] = &start_nonce;
    cd->arg[1] = &cd->device_data[0];
    cd->arg[2] = &name_dev;  // ponteiro device
    cd->arg[3] = &name_len;

    // lançar kernel
    lauch_kernel(cd);
    synchronize_cuda(cd);

    // copiar resultados de volta
    device_to_host_copy(cd, 0);

    u32_t used = coins_host[0];
    u32_t n_coins = (used - 1) / 14;

    printf("Kernel run : %u coin(s) found\n", n_coins);

    // guardar coins encontradas
    for (u32_t i = 0; i < n_coins; i++)
        save_coin(&coins_host[1 + i * 14]);
    save_coin(NULL); // flush

    // libertar memória device do nome
    if (name_len > 0) CU_CALL(cuMemFree, (name_dev));
}

//
// main()
//
int main(int argc, char *argv[])
{
    double time_limit = 0.0; // 0 = sem limite
    const char *name = ""; // default
    u08_t name_len;

    if (argc > 1) {
        name = argv[1];               // primeiro argumento: nome do minerador
        name_len = (u08_t)strlen(name);
    } else {
        name_len = (u08_t)strlen(name);
        printf("No miner name given. Using default: %s\n", name);
    }

    if (argc > 2) {
        time_limit = atof(argv[2]);   // segundo argumento opcional: tempo limite
        if (time_limit <= 0.0) {
            printf("Invalid time value. Usage: %s [miner_name] [time_in_seconds]\n", argv[0]);
            return 1;
        }
        printf("Running for %.2f seconds...\n", time_limit);
    } else {
        printf("No time limit — mining indefinitely.\n");
    }

    const int n_blocks = 512;
    u64_t start = 0;
    u64_t total_hashes = 0;

    printf("=== DETI Coin CUDA Miner ===\n");

    cuda_data_t cd;
    cd.device_number = 0;
    cd.cubin_file_name = "deti_miner_kernel.cubin";
    cd.kernel_name = "deti_miner_kernel";
    cd.n_kernel_arguments = 4;
    cd.data_size[0] = COINS_BUFFER_WORDS * sizeof(u32_t);
    cd.data_size[1] = 0;

    initialize_cuda(&cd);

    // medir tempo total
    time_measurement();
    double elapsed = 0.0;

    for (int batch = 0; batch < 10000; batch++) {
        mine_deti_coins(start, n_blocks, &cd, name, name_len);

        u64_t hashes_this_batch = (u64_t)n_blocks * THREADS_PER_BLOCK * ITER_PER_THREAD;
        total_hashes += hashes_this_batch;
        start += hashes_this_batch;

        time_measurement();
        elapsed += wall_time_delta();

        printf("Tempo total decorrido: %.2f s | Hashes: %llu\n",
               elapsed, (unsigned long long)total_hashes);

        if (time_limit > 0.0 && elapsed >= time_limit) {
            printf("\n=== Tempo limite atingido (%.2f / %.2f) ===\n", elapsed, time_limit);
            printf("Tentativas totais: %llu\n", (unsigned long long)total_hashes);
            printf("Taxa média: %.2f Mhash/s\n", (total_hashes / elapsed) / 1e6);
            break;
        }

        ((u32_t *)cd.host_data[0])[0] = 1;
        host_to_device_copy(&cd, 0);
    }

    terminate_cuda(&cd);

    printf("Mineração concluída.\n");
    return 0;
}

