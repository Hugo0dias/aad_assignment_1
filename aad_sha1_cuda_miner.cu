// sha1_cuda_miner.cu
// Minerador CUDA para DETI coins 2025 (SHA1, 55 bytes, template "DETI coin 2 ")

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <string.h>

#include "aad_data_types.h"   // u32_t, u08_t
#include "aad_sha1.h"         // CUSTOM_SHA1_CODE()
#include "aad_sha1_cpu.h"     // opcional: para validação no host

// tweak these for your GPU
#define THREADS_PER_BLOCK 256
#define BLOCKS 1024
#define ITER_PER_THREAD 2048     // quantos nonces cada thread testa por lançamento
#define MAX_COINS_PER_BATCH 2048 // espaço (em coins) guardado por batch (host/device must match)

//
// device SHA1 implementation (uses CUSTOM_SHA1_CODE)
// expects data as u32_t data[14] representing 56 bytes with last byte = 0x80
//
__device__ void sha1_device(u32_t *data, u32_t *hash)
{
#define T            u32_t
#define C(c)         (c)
#define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
#define DATA(idx)    data[idx]
#define HASH(idx)    hash[idx]
  CUSTOM_SHA1_CODE();
#undef T
#undef C
#undef ROTATE
#undef DATA
#undef HASH
}

//
// Kernel: cada thread gera mensagens que seguem o template, altera bytes 12..53 (nonce),
// testa ITER_PER_THREAD nonces espaçados por gridSize, e guarda coins encontrados.
//
extern "C" __global__ void deti_coins_sha1_search(
    u32_t *d_storage_area_words, // buffer onde serão gravadas as coins (in words); espaço deve ser >= 1 + MAX_COINS*14 if using first word as counter OR use d_count separate
    u32_t *d_hash_area_words,    // optional: store hashes (5 words per coin) or NULL
    unsigned long long start_nonce,
    int *d_count)                // atomic counter (number of coins found in this batch)
{
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int gridSize = gridDim.x * blockDim.x;

    // local coin buffer (14 words -> 56 bytes)
    u32_t coin[14];
    u08_t *msg = (u08_t*)coin;

    // prepare template once: zero + template + newline + 0x80 padding (applied with XOR-3)
    // Note: use coin_template and apply XOR 3 as in CPU version
    memset(msg, 0, 56);
    const char coin_template[] = "DETI coin 2 "; // 12 chars (no trailing '\0' in message)
    for (int i = 0; i < 12; ++i) msg[i ^ 3] = (u08_t)coin_template[i];
    msg[54 ^ 3] = (u08_t)'\n';
    msg[55 ^ 3] = (u08_t)0x80;

    // each thread will test ITER_PER_THREAD nonces:
    // base nonce for this gid:
    unsigned long long base = start_nonce + (unsigned long long)gid;

    for (int iter = 0; iter < ITER_PER_THREAD; ++iter) {
        unsigned long long nonce = base + (unsigned long long)iter * (unsigned long long)gridSize;

        // write nonce into bytes 12..53 (LSB first), preserving template bytes and padding
        unsigned long long tmp = nonce;
        for (int i = 12; i < 54; ++i) {
            msg[i ^ 3] = (u08_t)(tmp & 0xFFu);
            tmp >>= 8;
        }

        // compute SHA1 of coin (55 bytes)
        u32_t hash[5];
        sha1_device(coin, hash);

        // check signature: first word must equal 0xAAD20250u
        if (hash[0] == 0xAAD20250u) {
            // optional: count trailing zero bits in remaining words (value)
            // but at minimum we store the coin
            int pos = atomicAdd(d_count, 1);
            if (pos < MAX_COINS_PER_BATCH) {
                // write coin (14 words) into storage area at pos
                int base_w = pos * 14;
                for (int w = 0; w < 14; ++w) d_storage_area_words[base_w + w] = coin[w];
                // optionally store hash too
                if (d_hash_area_words) {
                    int base_h = pos * 5;
                    for (int h = 0; h < 5; ++h) d_hash_area_words[base_h + h] = hash[h];
                }
            }
            // continue searching (could break to avoid multiple writes from same thread)
        }
    }
}

// ---------------- Host code (main) ----------------
int main(int argc, char **argv)
{
    // device buffers
    u32_t *d_storage = NULL; // stores up to MAX_COINS_PER_BATCH coins (each coin=14 words)
    u32_t *d_hashes = NULL;  // optional: store hashes (5 words per coin)
    int *d_count = NULL;
    int h_count = 0;

    size_t storage_words = (size_t)MAX_COINS_PER_BATCH * 14u;
    size_t hash_words = (size_t)MAX_COINS_PER_BATCH * 5u;

    cudaMalloc((void**)&d_storage, storage_words * sizeof(u32_t));
    cudaMalloc((void**)&d_hashes, hash_words * sizeof(u32_t));
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    u32_t *h_storage = (u32_t*)malloc(storage_words * sizeof(u32_t));
    // u32_t *h_hashes = (u32_t*)malloc(hash_words * sizeof(u32_t));

    unsigned long long start_nonce = 0ULL;

    // timing events
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    FILE *fp = fopen("deti_coins_cuda_sha1.txt", "ab");
    if (!fp) { perror("fopen"); return 1; }

    const unsigned long long tries_per_batch = (unsigned long long)BLOCKS * (unsigned long long)THREADS_PER_BLOCK * (unsigned long long)ITER_PER_THREAD;

    while (1) {
        // reset device counter
        cudaMemset(d_count, 0, sizeof(int));

        // record and launch
        cudaEventRecord(ev_start, 0);
        deti_coins_sha1_search<<<BLOCKS, THREADS_PER_BLOCK>>>(d_storage, d_hashes, start_nonce, d_count);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
            break;
        }
        // record end then wait
        cudaEventRecord(ev_end, 0);
        cudaEventSynchronize(ev_end);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_end);

        // get how many coins found this batch
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_count > 0) {
            if (h_count > MAX_COINS_PER_BATCH) h_count = MAX_COINS_PER_BATCH;
            cudaMemcpy(h_storage, d_storage, (size_t)h_count * 14u * sizeof(u32_t), cudaMemcpyDeviceToHost);
	    
            // write each coin as 56 bytes to file (apply XOR-3 ordering when reading)
            for (int i = 0; i < h_count; ++i) {
                u08_t buf[56];
                u32_t *cw = &h_storage[i * 14];
                u08_t *msg = (u08_t*)cw;
                // Note: when device wrote coin as u32 words, the underlying byte ordering matches sha1_device expectations.
                // We simply write the 56 bytes to file
		// desfaz o XOR-3 para cada byte
		for (int j = 0; j < 56; ++j) {
		    buf[j] = msg[j ^ 3];
		}
                fwrite(buf, 1, 56, fp);
                fflush(fp);
                // optional: print hash stored (copy hashes if needed)
            }
            printf("Saved %d coins from this batch to deti_coins_cuda_sha1.txt\n", h_count);
        }

        double seconds = (double)ms / 1000.0;
        double mhps = (seconds > 0.0) ? ((double)tries_per_batch / 1.0e6) / seconds : 0.0;
        printf("Batch start_nonce=%llu : tried %llu hashes in %.3f s -> %.3f Mhash/s ; found %d\n",
               start_nonce, (unsigned long long)tries_per_batch, seconds, mhps, h_count);

        // advance nonce for next batch
        start_nonce += (unsigned long long)BLOCKS * (unsigned long long)THREADS_PER_BLOCK * (unsigned long long)ITER_PER_THREAD;
    }

    // cleanup (never reached in infinite loop)
    fclose(fp);
    free(h_storage);
    cudaFree(d_storage);
    cudaFree(d_hashes);
    cudaFree(d_count);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    return 0;
}
