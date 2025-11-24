/* client.c - minimal client that asks for (start,len) and mines that range.
   Adapta sha1 invocation de acordo com o teu projecto (sha1_cpu). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include "../aad_data_types.h"   // include primeiro
#include "../aad_sha1_cpu.h"   // se o header estiver no diretório acima
#include "../aad_sha1.h"   // se o header estiver no diretório acima

#include "x_common.h"


#define MAX_MESSAGE_SIZE 20u



/* connect */
static int connect_to_server(const char *ip, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { perror("socket"); exit(1); }
  struct sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = inet_addr(ip);
  sa.sin_port = htons(port);
  if (connect(fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) { perror("connect"); exit(1); }
  return fd;
}

/* simple bswap32 */
static inline uint32_t bswap32_u32(uint32_t x) { return ((x>>24)&0xFF) | ((x>>8)&0xFF00) | ((x<<8)&0xFF0000) | ((x<<24)&0xFF000000); }

/* mapping counter -> nonce bytes (minimal & deterministic):
   place counter little-endian in first 8 bytes of nonce field, rest zeros.
   YOU CAN REPLACE WITH TUA função textual para gerar nonce "DETI coin 2 <nonce_text>".
*/
/* Converte um counter para um nonce de 42 bytes,
   onde cada byte está entre 0x20 e 0x9F (160 valores possíveis).
*/
static void counter_to_nonce(uint64_t counter, uint8_t nonce42[42]) {
    const uint8_t MIN = 0x20;
    const uint8_t MAX = 0x9F;
    const uint32_t RANGE = (MAX - MIN + 1); // 160

    /* preenche tudo com MIN (0x20) */
    for (int i = 0; i < 42; i++)
        nonce42[i] = MIN;

    /* converter counter em "base 160" */
    int pos = 41;
    uint64_t x = counter;

    while (x > 0 && pos >= 0) {
        uint64_t digit = x % RANGE;     // valor entre 0 e 159
        nonce42[pos] = MIN + digit;     // força intervalo 0x20..0x9F
        x /= RANGE;
        pos--;
    }
}


/* build msg bytes (56 bytes) from template + nonce42 */
static void build_msg_from_nonce(uint8_t msg56[56], const uint8_t nonce42[42]) {
  const char templ[12] = "DETI coin 2 ";
  /* msg layout: bytes 0..11 template, 12..53 nonce (42 bytes), 54 = '\n', 55 = 0x80 */
  for (int i = 0; i < 12; ++i) msg56[i] = (uint8_t)templ[i];
  for (int i = 0; i < 42; ++i) msg56[12 + i] = nonce42[i];
  msg56[54] = '\n';
  msg56[55] = 0x80;

  // printf("msg56: ");
  // for (int i = 0; i < 56; ++i) printf("%02X", msg56[i]);
  // printf("\n");
}

/* Assumed sha1 function prototype:
   void sha1_cpu(const uint32_t coin_words[14], uint32_t hash_out[5]);
   If o teu projecto tem outro nome, adapta aqui.
*/

int main(int argc, char **argv) {


  if (argc != 3) {
    fprintf(stderr,"usage: %s server-ip port\n", argv[0]);
    return 1;
  }
  const char *server_ip = argv[1];
  int port = atoi(argv[2]);

  while (1) {
    int sock = connect_to_server(server_ip, port);
    /* ask for work: send empty message */
    message_t req; req.message_size = 0;
    if (send_message(sock, &req) < 0) { close(sock); sleep(1); continue; }

    /* expect (start_low, start_high, length) */
    message_t m;
    if (receive_message(sock, &m) < 0) { close(sock); sleep(1); continue; }
    if (m.message_size == 0) { /* no work */ close(sock); break; }
    if (m.message_size != 3) { close(sock); sleep(1); continue; }

    uint32_t start_low = m.data[0];
    uint32_t start_high = m.data[1];
    uint32_t length = m.data[2];
    uint64_t start = ((uint64_t)start_high << 32) | start_low;
    uint64_t end = start + (uint64_t)length;

    printf("got work start=%llu len=%u\n", (unsigned long long)start, length);

    /* iterate range and check coins */
    for (uint64_t counter = start; counter < end; ++counter) {
      // printf("testing counter %llu\r", (unsigned long long)counter); fflush(stdout);
      uint8_t nonce42[42];
      counter_to_nonce(counter, nonce42);

      uint8_t msg56[56];
      build_msg_from_nonce(msg56, nonce42);

      /* build coin words (14 u32) in the expected big-endian layout */
      uint32_t coin_words[14];
      for (int i = 0; i < 14; ++i) {
        uint32_t w = 0;
        /* read 4 bytes as they are in msg56, but coin_words should be big-endian word */
        for (int b = 0; b < 4; ++b) {
          int idx = i*4 + b;
          uint8_t v = (idx < 56) ? msg56[idx] : 0;
          w |= ((uint32_t)v) << (8*b);
        }
        /* the SHA1 routines in your project expect words in big-endian order.
           if they expect little-endian adapt this. Here we use bswap to convert
           the little-endian constructed w into big-endian layout. */
        coin_words[i] = bswap32_u32(w);   // sem byte-swap
        // if (counter > 1000000000) {
        //   printf("coin_words[%d] = %08X\n", i, coin_words[i]);
        //   printf("\n");
        // }
      }

      u32_t hash[5];
      /* call your SHA1: adapt name/signature if needed */
      sha1(coin_words, hash); /* <-- adapta isto conforme a tua API */

      // printf("hash[0]: %08X\n", hash[0]);


      if (hash[0] == 0xAAD20250u) {
        /* found coin -> send to server (14 words) */
        printf("found coin for counter %llu: ", (unsigned long long)counter);
        for (int i = 0; i < 14; ++i) printf("%08X", coin_words[i]);
        printf("\n");
        int sock2 = connect_to_server(server_ip, port);
        message_t out;
        out.message_size = 14;
        for (int i = 0; i < 14; ++i) out.data[i] = coin_words[i];
        /* enviar coin em NOVA ligação */
        if (send_message(sock2, &out) < 0) {
            fprintf(stderr,"error sending coin to server\n");
            perror("send_message failed");
        } else {
            printf("sent coin for counter %llu\n", (unsigned long long)counter);
        }
        
        close(sock2);
        
      

        /* optionally continue searching same range */
      }
    } /* end for counter */

    // NÃO fechar aqui! Só quando cliente terminar ou enviar coin

    continue;  

    /* after finishing chunk, connect again and request more */
  } /* end while */

  return 0;
}
