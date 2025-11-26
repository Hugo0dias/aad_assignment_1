/* server.c - minimal server that hands out 64-bit start + length ranges
   and accepts coins (14 words). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <errno.h>
#include <stdint.h>

#include "../aad_data_types.h"   // include primeiro
#include "../aad_sha1_cpu.h"   // se o header estiver no diretório acima
#include "../aad_sha1.h"   // se o header estiver no diretório acima
#include "../aad_utilities.h"
#include "../aad_vault.h"


#define MAX_MESSAGE_SIZE 20u

typedef struct {
  unsigned int message_size;
  unsigned int data[MAX_MESSAGE_SIZE];
} message_t;

/* I/O helpers (blocking + timeout) */
static int write_data(int socket_fd, void *data, size_t data_size) {
  struct timeval time_out;
  fd_set w_mask, e_mask;
  ssize_t bytes_written;

  unsigned char *ptr = data;
  while (data_size > 0) {
    FD_ZERO(&w_mask); FD_SET(socket_fd, &w_mask);
    FD_ZERO(&e_mask); FD_SET(socket_fd, &e_mask);
    time_out.tv_sec = 10; time_out.tv_usec = 0;
    if (select(socket_fd + 1, NULL, &w_mask, &e_mask, &time_out) <= 0) return -1;
    if (!FD_ISSET(socket_fd, &w_mask)) return -1;
    bytes_written = send(socket_fd, ptr, data_size, 0);
    if (bytes_written <= 0) return -2;
    ptr += bytes_written;
    data_size -= bytes_written;
  }
  return 0;
}

static int read_data(int socket_fd, void *data, size_t data_size) {
  struct timeval time_out;
  fd_set r_mask, e_mask;
  ssize_t bytes_read;
  unsigned char *ptr = data;

  while (data_size > 0) {
    FD_ZERO(&r_mask); FD_SET(socket_fd, &r_mask);
    FD_ZERO(&e_mask); FD_SET(socket_fd, &e_mask);
    time_out.tv_sec = 10; time_out.tv_usec = 0;
    if (select(socket_fd + 1, &r_mask, NULL, &e_mask, &time_out) <= 0) return -1;
    if (!FD_ISSET(socket_fd, &r_mask)) return -1;
    bytes_read = recv(socket_fd, ptr, data_size, 0);
    if (bytes_read <= 0) return -2;
    ptr += bytes_read;
    data_size -= bytes_read;
  }
  return 0;
}

/* send/receive message with htonl/ntohl conversions */
static int send_message(int socket_fd, message_t *m) {
  if (!m || m->message_size > MAX_MESSAGE_SIZE) return -10;
  unsigned int s = m->message_size;
  unsigned int tmp_size = htonl(m->message_size);
  /* make local copy to convert data without modifying caller's buffer */
  unsigned int buf[1 + MAX_MESSAGE_SIZE];
  buf[0] = tmp_size;
  for (unsigned int i = 0; i < s; ++i) buf[1 + i] = htonl(m->data[i]);
  int r = write_data(socket_fd, buf, (size_t)(1 + s) * sizeof(unsigned int));
  return r;
}

static int receive_message(int socket_fd, message_t *m) {
  if (!m) return -10;
  int r = read_data(socket_fd, &m->message_size, sizeof(unsigned int));
  if (r < 0) return r;
  unsigned int s = ntohl(m->message_size);
  if (s > MAX_MESSAGE_SIZE) return -11;
  m->message_size = s;
  if (s > 0) {
    r = read_data(socket_fd, &m->data[0], (size_t)s * sizeof(unsigned int));
    if (r < 0) return r;
    for (unsigned int i = 0; i < s; ++i) m->data[i] = ntohl(m->data[i]);
  }
  return 0;
}

static void close_socket(int fd) {
  close(fd);
}

static int setup_server(int port) {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) { perror("socket"); exit(1); }
  struct sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_ANY);
  sa.sin_port = htons(port);
  if (bind(listen_fd, (struct sockaddr*)&sa, sizeof(sa)) < 0) { perror("bind"); exit(1); }
  if (listen(listen_fd, 8) < 0) { perror("listen"); exit(1); }
  return listen_fd;
}

static int get_connection(int listen_fd, char ipbuf[32]) {
  struct sockaddr_in peer;
  socklen_t plen = sizeof(peer);
  int conn;
  do { conn = accept(listen_fd, (struct sockaddr*)&peer, &plen); } while (conn < 0 && errno == EINTR);
  if (conn < 0) { perror("accept"); return -1; }
  if (ipbuf) snprintf(ipbuf, 32, "%s", inet_ntoa(peer.sin_addr));
  return conn;
}


int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr,"usage: %s port chunk_size\n", argv[0]);
    fprintf(stderr,"   port: 49152..65535\n");
    fprintf(stderr,"   chunk_size: how many nonces per allocation (u32)\n");
    return 1;
  }
  int port = atoi(argv[1]);
  uint32_t chunk = (uint32_t)atoi(argv[2]);
  if (port < 49152 || port > 65535) { fprintf(stderr,"port out of range\n"); return 1; }
  if (chunk == 0) { fprintf(stderr,"chunk must be >0\n"); return 1; }

  int listen_fd = setup_server(port);
  printf("server listening on port %d, chunk=%u\n", port, chunk);

  uint64_t next_start = 0; // next 64-bit counter to give out

  while (1) {
    char ip[32];
    int conn = get_connection(listen_fd, ip);
    if (conn < 0) continue;
    printf("client connected: %s\n", ip);

    message_t m;
    if (receive_message(conn, &m) < 0) {
      fprintf(stderr,"recv error from client\n");
      close_socket(conn);
      continue;
    }

    if (m.message_size == 0) {
      /* client requests work: send start_low, start_high, length */
      uint32_t start_low = (uint32_t)(next_start & 0xFFFFFFFFu);
      uint32_t start_high = (uint32_t)(next_start >> 32);
      message_t out;
      out.message_size = 3;
      out.data[0] = start_low;
      out.data[1] = start_high;
      out.data[2] = chunk;
      if (send_message(conn, &out) < 0) fprintf(stderr,"send error\n");
      printf("gave work start=%llu len=%u\n", (unsigned long long)next_start, chunk);
      next_start += (uint64_t)chunk;
    }
    else if (m.message_size == 14) {
      /* client reports a coin: 14 u32 words */
      printf("received coin from client %s : ", ip);
      unsigned int coin[14];
      for (int i = 0; i < 14; ++i) { coin[i] = m.data[i]; printf("%08X", coin[i]); }
      printf("\n");
      save_coin(coin);
      save_coin(NULL);
      printf("coin saved to vault.\n");

      /* after receiving coin we still give new work: send next range */
      uint32_t start_low = (uint32_t)(next_start & 0xFFFFFFFFu);
      uint32_t start_high = (uint32_t)(next_start >> 32);
      message_t out;
      out.message_size = 3;
      out.data[0] = start_low;
      out.data[1] = start_high;
      out.data[2] = chunk;
      if (send_message(conn, &out) < 0) fprintf(stderr,"send error\n");
      printf("gave work start=%llu len=%u\n", (unsigned long long)next_start, chunk);
      next_start += (uint64_t)chunk;
    }
    else {
      fprintf(stderr,"unexpected message size %u\n", m.message_size);
      /* reply empty -> tell client no work */
      message_t out = { .message_size = 0 };
      send_message(conn, &out);
    }

    close_socket(conn);
  }

  close_socket(listen_fd);
  return 0;
}
