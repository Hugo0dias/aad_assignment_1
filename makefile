#
# Arquiteturas de Alto Desempenho 2025/2026
#
# makefile for the first practical assignment (A1)
#
# makefile automatic variables:
#   $@ is the name of the target
#   $< is the name of the first prerequisite
#   $^ is the list of names of all prerequisites (without duplicates)
#

#
# CUDA installation directory --- /usr/local/cuda or $(CUDA_HOME)
#

CUDA_DIR = /usr/local/cuda


#
# OpenCL installation directory (for a NVidia graphics card, sama as CUDA)
#

OPENCL_DIR = $(CUDA_DIR)


#
# CUDA device architecture
#
#   GeForce GTX 1660 Ti --- sm_75
#   RTX A2000 Ada --------- sm_86
#   RTX A6000 Ada --------- sm_86
#   RTX 4070 -------------- sm_89
#

CUDA_ARCH = sm_75


#
# clean up
#

clean:
	rm -f sha1_tests
	rm -f sha1_cuda_test sha1_cuda_kernel.cubin
	rm -f a.out
	rm -f cpu_miner
	rm -f sha1_tests sha1_cuda_test sha1_cuda_kernel.cubin cuda_miner
	rm -f deti_miner deti_miner.o deti_miner_kernel.cubin
	rm -f *.o a.out


#
# test the CUSTOM_SHA1_CODE macro
#

sha1_tests:	aad_sha1_cpu_tests.c aad_sha1.h aad_data_types.h aad_utilities.h makefile
	cc -march=native -Wall -Wshadow -Werror -O3 $< -o $@

sha1_miner: aad_sha1_cpu_miner.c
	cc -O3 -march=native -funroll-loops -flto -fomit-frame-pointer -Wall aad_sha1_cpu_miner.c -o cpu_miner

sha1_miner_avx: aad_sha1_cpu_miner_avx.c
	gcc -O3 -march=native -funroll-loops -ffast-math -fomit-frame-pointer -mtune=native aad_sha1_cpu_miner_avx.c -o cpu_miner_avx

sha1_miner_utf8: aad_sha1_cpu_miner_utf8.c
	cc -O3 -march=native -funroll-loops -flto -fomit-frame-pointer -Wall aad_sha1_cpu_miner_utf8.c -o cpu_miner_utf8

sha1_miner_avx_utf8: aad_sha1_cpu_miner_avx_utf8.c
	gcc -O3 -march=native -funroll-loops -ffast-math -fomit-frame-pointer -mtune=native aad_sha1_cpu_miner_avx_utf8.c -o cpu_miner_avx_utf8

#avx_correct: avx_correction.c
#	gcc -O3 -march=native -funroll-loops -ffast-math -fomit-frame-pointer -mtune=native avx_correction.c -o avx_correct

sha1_cuda_test:	aad_sha1_cuda_test.c sha1_cuda_kernel.cubin aad_sha1.h aad_data_types.h aad_utilities.h aad_cuda_utilities.h makefile
	cc -march=native -Wall -Wshadow -Werror -O3 $< -o $@ -lcuda

sha1_cuda_miner: aad_sha1_cuda_miner.cu
	nvcc -O3 -arch=sm_75 -o cuda_miner aad_sha1_cuda_miner.cu aad_sha1_cuda_kernel.cu -I.

# nome do binário final
all: deti_miner

# compilar o host program
deti_miner: deti_miner.o deti_miner_kernel.cubin
	$(CC) -o $@ deti_miner.o -lcuda -lcudart

# compilar o código C do host
deti_miner.o: deti_miner.c aad_data_types.h aad_utilities.h aad_vault.h aad_cuda_utilities.h
	$(CC) -c $< -O3 -o $@

# compilar o kernel CUDA para .cubin
deti_miner_kernel.cubin: deti_miner_kernel.cu aad_sha1.h aad_data_types.h
	nvcc -arch=$(CUDA_ARCH) --compiler-options -O2,-Wall -I$(CUDA_DIR)/include --cubin $< -o $@

