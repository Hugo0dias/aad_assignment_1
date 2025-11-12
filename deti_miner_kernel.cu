//
// deti_miner_kernel.cu
// Arquiteturas de Alto Desempenho 2025/2026
// Kernel CUDA para mineração DETI coins (mensagens geradas "on the fly")
// Cada thread cria e testa mensagens únicas, minimizando transferências host↔device.
//

#include "aad_data_types.h"
#include "aad_sha1.h"

#define COINS_BUFFER_WORDS 1024  // buffer comum host/device
#define ITER_PER_THREAD 2048     // quantos nonces cada thread testa

//
// Gera mensagem “DETI coin 2 …” com base num identificador único (thread + offset)
//
__device__ void generate_message_from_id(u64_t nonce, u32_t *data)
{
    // Temporary buffer for message
    u08_t msg[56];
    
    // 1. Write template with XOR 3 to match host byte order
    const char prefix[] = "DETI coin 2 ";
    for (int i = 0; i < 12; ++i) {
        msg[i ^ 3] = (u08_t)prefix[i];
    }

    // 2. Write nonce bytes with XOR 3
    unsigned long long tmp = nonce;
    for (int i = 12; i < 54; ++i) {
        u08_t b = (u08_t)(tmp & 0xFFu);         // get the least significant byte (8 bits)
        if (b == (u08_t)'\n') b = (u08_t)'\b';  // avoid newline (\n), substitute with backspace (\bs)
        msg[i ^ 3] = b;
        tmp >>= 8;                              // move 1 byte to the right
    }

    // 3. Add terminator bytes with XOR 3
    msg[54 ^ 3] = (u08_t)'\n';
    msg[55 ^ 3] = (u08_t)0x80;

    // 4. Copy bytes directly to output words
    // No need for additional byte reordering since msg[] is already in XOR 3 order
    memcpy(data, msg, 56);
}

// funçao kernel chamada pelo host (cpu)
// extern "C" para evitar name mangling do C++ (encontrar a funçao pelo nome)
extern "C" __global__ void deti_miner_kernel(u64_t start_nonce, u32_t *coins_storage_area)
{   
    // definidas automaticamente pelo CUDA (1o valor é 0)
    const u64_t gid = (u64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes a contiguous block of nonces
    u64_t nonce_base = start_nonce + gid * ITER_PER_THREAD;
    
    u32_t data[14];
    u32_t hash[5];
    // vai haver max|gid_value| threads a fazer ITER_PER_THREAD iterações cada
    for (int iter = 0; iter < ITER_PER_THREAD; ++iter) {
        u64_t nonce = nonce_base + iter;
        
        // Generate message with correct byte order
        generate_message_from_id(nonce, data);

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