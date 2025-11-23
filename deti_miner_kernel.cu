/*
//
// Cada thread cria e testa mensagens únicas, minimizando transferências host↔device.
// Código ALL BYTES

#include "aad_data_types.h"
#include "aad_sha1.h"
#include <cstdio>

#define COINS_BUFFER_WORDS 1024  // buffer comum host/device
#define ITER_PER_THREAD 2048     // quantos nonces cada thread testa

//
// Gera mensagem “DETI coin 2 …” com base num identificador único (thread + offset)
//
__device__ void generate_message_from_id(u64_t nonce, u32_t *data, const char *name, u08_t name_len)
{
    // Temporary buffer for message
    u08_t msg[56] = {0};
    
    // 1. Write template with XOR 3 to match host byte order
    const char prefix[] = "DETI coin 2 ";
    for (int i = 0; i < 12; ++i) {
        msg[i ^ 3] = (u08_t)prefix[i];
    }

    if (name_len > 0) {
        for (int i = 0; i < name_len && i < (54 - 12); ++i) { 
            // limite: não ultrapassar buffer antes do nonce
            msg[(12 + i) ^ 3] = (u08_t)name[i];
        }
    }

    // 2. Escrever nonce após o prefixo + 2 bytes + name
    unsigned long long tmp = nonce;
    int nonce_start = 12 + name_len;
    int nonce_end = 54; // byte 54 reservado para '\n'
    for (int i = nonce_start; i < nonce_end; ++i) {
        u08_t b = (u08_t)(tmp & 0xFFu);
        if (b == (u08_t)'\n') b = (u08_t)'\b'; // evitar newline
        msg[i ^ 3] = b;
        tmp >>= 8;
    }

    // 3. Add terminator bytes with XOR 3
    msg[54 ^ 3] = (u08_t)'\n';
    msg[55 ^ 3] = (u08_t)0x80;

    // 4. Copy bytes directly to output words
    // No need for additional byte reordering since msg[] is already in XOR 3 order
    memcpy(data, msg, 56);

    //const u64_t gid = (u64_t)blockIdx.x * blockDim.x + threadIdx.x;
    //if (gid == 0) {
    //    printf("Thread 0 first 5 coins bytes in hex:\n");
    //    for (int i = 0; i < 56; ++i) {
    //        printf("%02x ", msg[i]);
    //        if ((i+1) % 16 == 0) printf("\n");
    //    }
    //    printf("\n");
    //}
}

// funçao kernel chamada pelo host (cpu)
// extern "C" para evitar name mangling do C++ (encontrar a funçao pelo nome)
extern "C" __global__ void deti_miner_kernel(u64_t start_nonce, u32_t *coins_storage_area, const char *name, u08_t name_len)
{   
    // definidas automaticamente pelo CUDA (1o valor é 0)
    
    //Bloco 0 -> threadIdx.x: 0 1 2 3 -> gid: 0 1 2 3
    //Bloco 1 -> threadIdx.x: 0 1 2 3 -> gid: 4 5 6 7
    //Bloco 2 -> threadIdx.x: 0 1 2 3 -> gid: 8 9 10 11
    
   // calcular thread global id : bloco_id * bloco_tamanho + thread_id_no_bloco
    const u64_t gid = (u64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    // gerar nonce inicial para cada thread
    
    //nonce inicial -> start_nonce: 0 -> gid:0 -> nonce:0
    //nonce inicial -> start_nonce: 0 -> gid:1 -> nonce:2048
    //nonce inicial -> start_nonce: 0 -> gid:2 -> nonce:4096
    
    u64_t nonce_base = start_nonce + gid * ITER_PER_THREAD;
    
    u32_t data[14];
    u32_t hash[5];
    // cada thread testa 2048 nonces
    for (int iter = 0; iter < ITER_PER_THREAD; ++iter) {
        u64_t nonce = nonce_base + iter;
        
        // Generate message
        generate_message_from_id(nonce, data, name, name_len);

        // Compute SHA1
        #define T u32_t
        #define C(c) (c)
        #define ROTATE(x,n) ((x << n) | (x >> (32 - n)))
        #define DATA(idx) (data[idx])
        #define HASH(idx) (hash[idx])
        CUSTOM_SHA1_CODE();
        #undef T
        #undef C
        #undef ROTATE
        #undef DATA
        #undef HASH

        if (hash[0] == 0xAAD20250u) {
            // avança 14 words no buffer (1coin ocupa 14 words)
            u32_t idx = atomicAdd(coins_storage_area, 14u);
            // verifica se tem espaço no buffer
            if (idx < COINS_BUFFER_WORDS - 14u) {
                // guarda coin no index certo, 14 * 4 = 56 bytes
                memcpy(&coins_storage_area[idx], data, 14 * sizeof(u32_t));
            }
        }
    }
}
*/
//
// Cada thread cria e testa mensagens únicas, minimizando transferências host↔device.
// Código RANDOM UTF8

#include "aad_data_types.h"
#include "aad_sha1.h"
#include <cstdio>

#define COINS_BUFFER_WORDS 1024  // buffer comum host/device
#define ITER_PER_THREAD 2048     // quantos nonces cada thread testa

#define UTF8_TABLE_LEN 95
__constant__ char UTF8_TABLE[95] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "!#$%&()*+,-./:<=>?@[\\]^_{|}~";

__device__ inline u64_t xorshift64(u64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}


//
// Gera mensagem “DETI coin 2 …” com base num identificador único (thread + offset)
//
__device__ void generate_message_from_id(u64_t nonce, u32_t *data, const char *name, u08_t name_len)
{
    // Temporary buffer for message
    u08_t msg[56];
    //#pragma unroll
    for (int i = 0; i < 56; i++) msg[i] = 0;

    
    // 1. Write template with XOR 3 to match host byte order
    const char prefix[] = "DETI coin 2 ";
    for (int i = 0; i < 12; ++i) {
        msg[i] = (u08_t)prefix[i];
    }

    if (name_len > 0) {
        for (int i = 0; i < name_len && i < (54 - 12); ++i) { 
            // limite: não ultrapassar buffer antes do nonce
            msg[(12 + i)] = (u08_t)name[i];
        }
    }

    // 2. Escrever nonce após o prefixo + 2 bytes + name
    // unsigned long long tmp = nonce;
    int nonce_start = 12 + name_len;
    int nonce_end = 54; // byte 54 reservado para '\n'
    u64_t seed = nonce;   // começa com o nonce base

    for (int i = nonce_start; i < nonce_end; ++i) {
        seed = xorshift64(seed);
        msg[i] = UTF8_TABLE[seed % UTF8_TABLE_LEN];
    }

    // 3. Add terminator bytes with XOR 3
    msg[54] = (u08_t)'\n';
    msg[55] = (u08_t)0x80;

    // 4. Copy bytes directly to output words
    // No need for additional byte reordering since msg[] is already in XOR 3 order
    // memcpy(data, msg, 56);
    u08_t *dst = (u08_t*)data;
    for (int i = 0; i < 56; i++)
        dst[i] = msg[i ^ 3];

    /*const u64_t gid = (u64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0) {
        printf("Thread 0 first 5 coins bytes in hex:\n");
        for (int i = 0; i < 56; ++i) {
            printf("%02x ", msg[i]);
            if ((i+1) % 16 == 0) printf("\n");
        }
        printf("\n");
    }*/
}

// funçao kernel chamada pelo host (cpu)
// extern "C" para evitar name mangling do C++ (encontrar a funçao pelo nome)
extern "C" __global__ void deti_miner_kernel(u64_t start_nonce, u32_t *coins_storage_area, const char *name, u08_t name_len)
{   
    // definidas automaticamente pelo CUDA (1o valor é 0)
    /*
    Bloco 0 -> threadIdx.x: 0 1 2 3 -> gid: 0 1 2 3
    Bloco 1 -> threadIdx.x: 0 1 2 3 -> gid: 4 5 6 7
    Bloco 2 -> threadIdx.x: 0 1 2 3 -> gid: 8 9 10 11
    */
   // calcular thread global id : bloco_id * bloco_tamanho + thread_id_no_bloco
    const u64_t gid = (u64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    // gerar nonce inicial para cada thread
    /*
    nonce inicial -> start_nonce: 0 -> gid:0 -> nonce:0
    nonce inicial -> start_nonce: 0 -> gid:1 -> nonce:2048
    nonce inicial -> start_nonce: 0 -> gid:2 -> nonce:4096
    */
    u64_t nonce_base = start_nonce + gid * ITER_PER_THREAD;
    
    u32_t data[14];
    u32_t hash[5];
    // cada thread testa 2048 nonces
    for (int iter = 0; iter < ITER_PER_THREAD; ++iter) {
        u64_t nonce = nonce_base + iter;
        
        // Generate message
        generate_message_from_id(nonce, data, name, name_len);

        // Compute SHA1
        #define T u32_t
        #define C(c) (c)
        #define ROTATE(x,n) ((x << n) | (x >> (32 - n)))
        #define DATA(idx) (data[idx])
        #define HASH(idx) (hash[idx])
        CUSTOM_SHA1_CODE();
        #undef T
        #undef C
        #undef ROTATE
        #undef DATA
        #undef HASH

        if (hash[0] == 0xAAD20250u) {
            // avança 14 words no buffer (1coin ocupa 14 words)
            u32_t idx = atomicAdd(coins_storage_area, 14u);
            // verifica se tem espaço no buffer
            if (idx < COINS_BUFFER_WORDS - 14u) {
                // guarda coin no index certo, 14 * 4 = 56 bytes
                memcpy(&coins_storage_area[idx], data, 14 * sizeof(u32_t));
            }
        }
    }
}
