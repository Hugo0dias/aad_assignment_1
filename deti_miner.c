//
// Cada kernel gera e testa mensagens inteiras na GPU (sem transferências de dados massivas)
//

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

extern void deti_miner_kernel(u64_t start_nonce, u32_t *coins_storage_area, const char *name, u08_t name_len);
//
// Função que lança o kernel e recolhe os resultados
//
static u32_t mine_deti_coins(u64_t start_nonce, int n_blocks, cuda_data_t *cd,
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

    return n_coins;
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

    #define MAX_KERNEL_RUNS 10000
    #define HIST_BINS 1000  // cada bin = 1 ms
    #define COINS_HIST_BINS 100

    int hist[HIST_BINS] = {0};
    int coins_hist[COINS_HIST_BINS] = {0};
    int coins_per_kernel[MAX_KERNEL_RUNS];
    double kernel_times[MAX_KERNEL_RUNS];
    int kernel_run_count = 0;

    for (int batch = 0; batch < 10000; batch++) {
        time_measurement();
        u32_t n_coins = mine_deti_coins(start, n_blocks, &cd, name, name_len);
        double dt = wall_time_delta();

        // armazenar dados
        coins_per_kernel[kernel_run_count] = n_coins;
        kernel_times[kernel_run_count] = dt;

        // atualizar histogramas
        int coins_bin = n_coins;
        if (coins_bin >= COINS_HIST_BINS) coins_bin = COINS_HIST_BINS - 1;
        coins_hist[coins_bin]++;

        int time_bin = (int)(dt * 1000.0);
        if (time_bin >= HIST_BINS) time_bin = HIST_BINS - 1;
        hist[time_bin]++;

        // somar ao total
        elapsed += dt;
        u64_t hashes_this_batch = (u64_t)n_blocks * THREADS_PER_BLOCK * ITER_PER_THREAD;
        total_hashes += hashes_this_batch;
        start += hashes_this_batch;

        kernel_run_count++;

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

    // construir histograma
    for (int i = 0; i < kernel_run_count; i++) {
        int bin = (int)(kernel_times[i] * 1000.0); // segundos → ms
        if (bin >= HIST_BINS) bin = HIST_BINS - 1;
        hist[bin]++;
    }

    // imprimir histograma
    printf("Histogram of kernel times (ms):\n");
    for (int i = 0; i < HIST_BINS; i++) {
        if (hist[i] > 0)
            printf("%3d ms: %d\n", i, hist[i]);
    }

    printf("Histogram of DETI coins per kernel:\n");
    for (int i = 0; i < COINS_HIST_BINS; i++) {
        if (coins_hist[i] > 0)
            printf("%2d coins: %d kernels\n", i, coins_hist[i]);
    }

    terminate_cuda(&cd);

    printf("Mineração concluída.\n");
    return 0;
}

