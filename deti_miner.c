//
// deti_miner.c
// Arquiteturas de Alto Desempenho 2025/2026
// Host program para mineração de DETI coins com CUDA.
// -> Cada kernel gera e testa mensagens inteiras na GPU (sem transferências de dados massivas)
// -> Apenas o pequeno buffer de resultados é copiado de volta.
//

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
    coins_host[0] = 1; // índice livre inicial
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

