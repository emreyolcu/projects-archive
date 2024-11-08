#ifndef DEFS_H_
#define DEFS_H_

#include <linux/limits.h>   /* PATH_MAX */
#include <linux/types.h>    /* pid_t */

#define LINE_MAX 80
#define HIST_MAX 10
#define BOOK_MAX 10

#define streq(s1, s2) (strcmp((s1), (s2)) == 0)

#define CHECK(expr, ret, str) do { \
    if ((expr) == ret) {           \
        perror(str);               \
        exit(EXIT_FAILURE);        \
    }                              \
} while (0)                        \

typedef struct {
    char commands[HIST_MAX][LINE_MAX];
    int count;
} history;

typedef struct {
    char command[PATH_MAX];
    int set;
    pid_t pid;  /* pid of the process that is sleeping */
} alarm_info;

/*
 * Shell Functions
 */
void init_shell(history *h, alarm_info *a, char bookmarks[][PATH_MAX]);
void read_input(char input[]);
char *prompt(int type);
void eval_input(const char input[], history *h, alarm_info *a,
                char bookmarks[][PATH_MAX]);
void parse_input(char cpy[], char *args[], int *bg);
void run_command(char *args[]);
int builtin(char *args[], history *h, alarm_info *a,
            char bookmarks[][PATH_MAX]);

/*
 * Built-in Commands
 */
    /* cd */
void cd(const char *arg);
    /* history */
void add_to_hist(const char input[], history *h);
void print_hist(const history *h);
void repeat_hist(const char input[], history *h, alarm_info *a,
                 char bookmarks[][PATH_MAX]);
    /* goodmorning */
void setalarm(char *args[], alarm_info *a);
int sleepduration(char *args[]);
void silence(alarm_info *a);
    /* processinfo */
void procinfo_ins(char *args[]);
void procinfo_rm(void);
    /* bookmark */
void create_bookmark(char *args[], char bookmarks[][PATH_MAX]);
void goto_bookmark(char *args[], char bookmarks[][PATH_MAX]);

#endif
