CC=gcc
CFLAGS=-Wall -Wextra

quantization: tensor.c
	$(CC) $(CFLAGS) -o tensor tensor.c
