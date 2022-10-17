BINARY_NAME = sgemm_benchmarking
CUDA_PATH   = /local/cuda-10.1/toolkit
CC			= $(CUDA_PATH)/bin/nvcc
CFLAGS		= -O2 -std=c++11
LDFLAGS		= -L$(CUDA_PATH)/lib64 -lcudart -lcublas
INCFLAGS	= -I$(CUDA_PATH)/include -Iinclude


SRC			= $(wildcard *.cu)
build : $(BINARY_NAME)

$(BINARY_NAME): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(SRC) -o $(BINARY_NAME)

clean:
	rm $(BINARY_NAME)