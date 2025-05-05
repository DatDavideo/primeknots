CC := gcc
CFLAGS := -Wall -Wextra -pedantic -Wshadow -Wconversion -Werror -O2 -g

SRC := $(wildcard *.c)
OBJ := $(SRC:.c=.o)
TARGET := knotdata

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
