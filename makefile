NAME := cuda
SRC_DIR := src
BUILD_DIR := build
INCLUDE_DIR := include
LIB_DIR := lib
TESTS_DIR := tests
BIN_DIR := bin

CC := clang-18
LINTER := clang-tidy-18
FORMATTER := clang-format-18

CFLAGS := -std=gnu17 -D _GNU_SOURCE -D __STDC_WANT_LIB_EXT1__ -Wall -Wextra -pedantic
LDFLAGS := -lm

$(NAME): format lint dir $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(BIN_DIR)/$@ $(patsub %, build/%, $(OBJS))

$(OBJS): dir
	@mkdir -p $(BUILD_DIR)/$(@D)
	@$(CC) $(CFLAGS) -o $(BUILD_DIR)/$@ -c $*.c

test: dir
	@$(CC) $(CFLAGS) -lcunit -o $(BIN_DIR)/$(NAME)_test $(TESTS_DIR)/*.c
	@$(BIN_DIR)/$(NAME)_test

