CFLAGS = -std=c99 -Wall -Wextra -pedantic -D_GNU_SOURCE

all: shell-rl

shell-rl: CFLAGS += -DREADLINE_
shell-rl: LDFLAGS = -lreadline
shell-rl: shell

shell: shell.c

clean:
	rm -f shell
