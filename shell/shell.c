#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <linux/limits.h>
#include <errno.h>
#include <sys/prctl.h>
#include <time.h>
#include <fcntl.h>

#ifdef READLINE_
#include <readline/readline.h>
#endif

#include "defs.h"

/* -1 when not loaded,
 * 0 when loaded without argument,
 * pid when loaded with argument */
static int procinfo_status = -1;

int main(void)
{
    char input[LINE_MAX];
    history history;
    alarm_info alarm;
    char bookmarks[BOOK_MAX][PATH_MAX];

    init_shell(&history, &alarm, bookmarks);
    atexit(procinfo_rm);    /* Remove the kernel module before exiting. */

    for (;;) {
        read_input(input);
        eval_input(input, &history, &alarm, bookmarks);
    }

    return 0;
}

void init_shell(history *h, alarm_info *a, char bookmarks[BOOK_MAX][PATH_MAX])
{
    int i;

    h->count = 0;
    a->set = 0;
    for (i = 0; i < BOOK_MAX; i++) {
        bookmarks[i][0] = '\0';
    }

}

void read_input(char input[])
{
    char *p;
    int type = 1;
#ifndef READLINE_
    char c;
    char *nl;

    p = prompt(type);
    printf("%s", p);
    fflush(stdout);

    fgets(input, LINE_MAX, stdin);

    if (ferror(stdin)) {
        perror("fgets");
        exit(1);
    }
    else if (feof(stdin)) {
        exit(0);
    }

    /* Exhaust the input buffer and remove the newline at the end. */
    nl = strchr(input, '\n');
    if (nl == NULL) {
        while ((c = getchar()) != '\n' && c != EOF) {
            ;
        }
    }
    else {
        *nl = '\0';
    }
#else
    char *line;

    p = prompt(type);

    line = readline(p);
    if (line == NULL) {
        exit(0);
    }
    else {
        strncpy(input, line, LINE_MAX);
        input[LINE_MAX - 1] = '\0';
    }
    free(line);
#endif

    free(p);
}

/* If type == 1, return a prompt displaying the current working directory,
 * otherwise, return "shell". */
char *prompt(int type)
{
    char *p;
    char *tmp;
    char cwd[PATH_MAX];
    char *home;
    int hlen, cwdlen;

    tmp = NULL;

    if (type == 0) {
        tmp = "shell";
    }
    else if (type == 1) {
        getcwd(cwd, PATH_MAX);
        home = getenv("HOME");

        hlen = strlen(home);

        /* Replace /home/<user> with ~. */
        if (strncmp(cwd, home, hlen) == 0) {
            cwdlen = strlen(cwd);
            cwd[0] = '~';
            memmove(cwd + 1, cwd + hlen, cwdlen - hlen + 1);
        }

        tmp = cwd;
    }

    p = malloc(PATH_MAX + 16);  /* 16 extra characters for ANSI color codes */
    sprintf(p, "\033[1m\033[34m%s > \033[0m", tmp);

    return p;
}

void eval_input(const char input[], history *h, alarm_info *a,
                char bookmarks[BOOK_MAX][PATH_MAX])
{
    char cpy[LINE_MAX];
    char *args[LINE_MAX/2 + 1];
    int bg;
    pid_t pid;

    if (input[0] == '!') {
        repeat_hist(input, h, a, bookmarks);
    }
    else {
        /* parse_input modifies its first argument, so we pass it a copy of the
         * input because we later need the original input to add to history. */
        strncpy(cpy, input, LINE_MAX);
        parse_input(cpy, args, &bg);

        if (args[0] != NULL) {
            add_to_hist(input, h);

            if (!builtin(args, h, a, bookmarks)) {
                CHECK(pid = fork(), -1, "fork");
                if (pid == 0) {
                    run_command(args);
                    _Exit(EXIT_FAILURE);
                }
                else {
                    if (!bg) {
                        while (waitpid(pid, NULL, 0) > 0) {
                            ;
                        }
                    }
                    else {
                        printf("[bg] %ld\n", (long)pid);
                    }
                }
            }
        }
    }
}

void parse_input(char cpy[], char *args[], int *bg)
{
    char *sep = " \t\n";
    char *arg;
    int i;

    *bg = 0;
    for (arg = strtok(cpy, sep), i = 0; arg; arg = strtok(NULL, sep), i++) {
        if (arg[0] == '&') {
            *bg = 1;
            break;
        }
        else {
            args[i] = arg;
        }
    }

    args[i] = NULL;
}

void run_command(char *args[])
{
    char *sep = ":";
    char *pvar, *tkn, *p;
    char path[PATH_MAX];

    if (strchr(args[0], '/')) {
        /* If the command includes a "/", we have a path. */
        p = args[0];
    }
    else {
        p = path;
        pvar = strdup(getenv("PATH"));

        for (tkn = strtok(pvar, sep); tkn; tkn = strtok(NULL, sep)) {
            strcpy(path, tkn);

            /* Paths in PATH environment variable may or may not be terminated
             * with a '/'. */
            if (path[strlen(path) - 1] != '/') {
                strcat(path, "/");
            }
            strcat(path, args[0]);

            if (access(path, F_OK) == 0) {
                break;
            }
        }

        free(pvar);
    }

    if (execv(p, args) == -1) {
        fprintf(stderr, "Command not found: %s\n", args[0]);
    }
}

int builtin(char *args[], history *h, alarm_info *a,
            char bookmarks[BOOK_MAX][PATH_MAX])
{
    if (streq(args[0], "exit")) {
        exit(0);
    }
    else if (streq(args[0], "cd")) {
        cd(args[1]);
    }
    else if (streq(args[0], "history")) {
        print_hist(h);
    }
    else if (streq(args[0], "goodmorning")) {
        setalarm(args, a);
    }
    else if (streq(args[0], "silence")) {
        silence(a);
    }
    else if (streq(args[0], "processinfo")) {
        procinfo_ins(args);
    }
    else if (streq(args[0], "bookmark")) {
        create_bookmark(args, bookmarks);
    }
    else if (streq(args[0], "go")) {
        goto_bookmark(args, bookmarks);
    }
    else {
        return 0;
    }

    return 1;
}

void cd(const char *arg)
{
    char cwd[PATH_MAX];
    char *home;
    errno = 0;

    home = getenv("HOME");

    if (arg == NULL) {
        chdir(home);
    }
    else if (arg[0] == '~') {
        strcpy(cwd, home);
        strcat(cwd, arg + 1);
        chdir(cwd);
    }
    else if (streq(arg, "-")) {
        strcpy(cwd, getenv("OLDPWD"));
        printf("%s\n", cwd);
        chdir(cwd);
    }
    else {
        chdir(arg);
    }

    if (errno) {
        fprintf(stderr, "cd: %s: %s\n", strerror(errno), arg);
    }
    else {
        setenv("OLDPWD", getenv("PWD"), 1);
        getcwd(cwd, sizeof cwd);
        setenv("PWD", cwd, 1);
    }
}

void add_to_hist(const char input[], history *h)
{
    strncpy(h->commands[h->count % HIST_MAX], input, LINE_MAX);
    h->count++;
}

void print_hist(const history *h)
{
    int c;

    for (c = h->count; (h->count - c < HIST_MAX) && (c > 0); c--) {
        printf("%5d  %s\n", c, h->commands[(c - 1) % HIST_MAX]);
    }
}

void repeat_hist(const char input[], history *h, alarm_info *a,
                 char bookmarks[BOOK_MAX][PATH_MAX])
{
    int i, count;
    char c;
    char buf[LINE_MAX];
    char *cmd;
    int cmdlen;

    count = h->count;
    if (count == 0) {
        fprintf(stderr, "No commands in history\n");
        return;
    }

    c = input[1];
    buf[0] = '\0';
    if (c == '!') {
        i = count;
        sscanf(input, "!!%[^\n]", buf);
    }
    else if (isdigit(c)) {
        sscanf(input, "!%d%[^\n]", &i, buf);
    }
    else {
        fprintf(stderr, "Wrong use of !\n");
        return;
    }

    if (i < 1 || i > count || i < count - 10) {
        fprintf(stderr, "No such command in history\n");
    }
    else {
        /* Concatenate any new input to before running the command. */
        cmd = h->commands[(i - 1) % HIST_MAX];
        cmdlen = strlen(cmd);
        memmove(buf + cmdlen, buf, LINE_MAX - cmdlen - 1);
        memcpy(buf, cmd, cmdlen);
        buf[LINE_MAX - 1] = '\0';

        printf("%s\n", buf);
        eval_input(buf, h, a, bookmarks);
    }
}

void setalarm(char *args[], alarm_info *a)
{
    char *play[5] = { "mpg123", "--loop", "-1" };
    pid_t pid;
    int duration;
    int devnull;

    if (args[1] == NULL || args[2] == NULL) {
        printf("Wrong number of arguments\n"
               "Usage: goodmorning <time> <file>\n");
        return;
    }

    if (a->set) {
        if (kill(a->pid, SIGKILL) == 0) {
            printf("Previous alarm canceled\n");
        }
        else {
            perror("kill");
            return;
        }
    }

    play[3] = args[2];

    /* a->command is the exact command that is used to play the music, we need
     * this later to silence the alarm using pkill. */
    sprintf(a->command, "%s %s %s %s", play[0], play[1], play[2], play[3]);

    duration = sleepduration(args);

    CHECK(pid = fork(), -1, "fork");
    if (pid == 0) {
        prctl(PR_SET_PDEATHSIG, SIGHUP);    /* child exits if parent does */
        devnull = open("/dev/null", O_WRONLY);
        sleep(duration);
        printf("\nEnter the command \"silence\" to silence the alarm\n");
        dup2(devnull, STDOUT_FILENO);       /* redirect stdout to /dev/null */
        dup2(devnull, STDERR_FILENO);       /* redirect stderr to /dev/null */
        run_command(play);
        _Exit(EXIT_FAILURE);
    }
    else {
        a->set = 1;
        a->pid = pid;
        printf("Alarm is set for %s\n", args[1]);
    }
}

/* Calculates the time in seconds between now and when the alarm will go off.
 * The method is as follows:
 *  - Get a struct tm corresponding to the current time.
 *  - Modify its hour, minute and second values to match the argument while
 *  keeping the date constant.
 *  - Convert this struct to calendar time in seconds.
 *  - If this time is later than now, the alarm will go off within the same day,
 *  so return the time in between. Otherwise, the alarm will go off the next
 *  day, so return the time in between subtracted from 24 hours. */
int sleepduration(char *args[])
{
    time_t now, then;
    int hour, minute;
    struct tm tm;

    memset(&tm, 0, sizeof tm);

    sscanf(args[1], "%d.%d", &hour, &minute);

    time(&now);

    tm = *localtime(&now);
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = 0;
    then = mktime(&tm);

    if (then > now) {
        return then - now;
    }
    else {
        return 86400 - (now - then);    /* 24 hours = 86400 seconds */
    }
}

void silence(alarm_info *a)
{
    /* "-fx" is passed as argument to pkill because:
     *      -f enables pattern matching using the full command,
     *      -x enables killing the command with the exact matching name,
     * making sure we do not accidentally kill any other processes */
    char *args[4] = { "pkill", "-fx" };
    pid_t pid;

    args[2] = a->command;
    args[3] = NULL;

    CHECK(pid = fork(), -1, "fork");
    if (pid == 0) {
        run_command(args);
        _Exit(EXIT_FAILURE);
    }
    else {
        a->set = 0;
    }
}

void procinfo_ins(char *args[])
{
    char *cmd[5] = { "sudo", "insmod", "procinfo/procinfo.ko" };
    char pid_arg[20] = "arg=";
    pid_t pid;
    pid_t arg;

    if (args[1] == NULL) {
        if (procinfo_status >= 0) {
            printf("processinfo module has already been loaded.\n");
            return;
        }
        else {
            procinfo_status = 0;
            cmd[3] = NULL;
        }
    }
    else {
        arg = (int)strtol(args[1], NULL, 10);
        if (procinfo_status == arg) {
            printf("processinfo module has already been loaded.\n");
            return;
        }
        else {
            procinfo_rm();
            procinfo_status = arg;
            strcat(pid_arg, args[1]);
            cmd[3] = pid_arg;
            cmd[4] = NULL;
        }
    }

    CHECK(pid = fork(), -1, "fork");
    if (pid == 0) {
        run_command(cmd);
        _Exit(EXIT_FAILURE);
    }
    else {
        while (waitpid(pid, NULL, 0) > 0) {
            ;
        }
    }
}

void procinfo_rm(void)
{
    char *cmd[5] = { "sudo", "rmmod", "procinfo" };
    pid_t pid;

    if (procinfo_status >= 0) {
        CHECK(pid = fork(), -1, "fork");
        if (pid == 0) {
            run_command(cmd);
            _Exit(EXIT_FAILURE);
        }
        else {
            while (waitpid(pid, NULL, 0) > 0) {
                ;
            }
        }
    }
}

void create_bookmark(char *args[], char bookmarks[BOOK_MAX][PATH_MAX])
{
    int index;
    char cwd[PATH_MAX];

    if (args[1] == NULL) {
        printf("Usage: bookmark <index>\n");
        return;
    }

    index = (int)strtol(args[1], NULL, 10);
    if (index > BOOK_MAX || index < 0) {
        fprintf(stderr, "Valid values for bookmark indices are: 0 - %d\n",
                BOOK_MAX - 1);
        return;
    }

    getcwd(cwd, PATH_MAX);
    strcpy(bookmarks[index], cwd);
}

void goto_bookmark(char *args[], char bookmarks[BOOK_MAX][PATH_MAX])
{
    int index;
    int i;

    if (args[1] == NULL) {
        for (i = 0; i < BOOK_MAX; i++) {
            if (bookmarks[i][0] != '\0') {
                printf("%5d  %s\n", i, bookmarks[i]);
            }
        }
    }
    else {
        index = (int)strtol(args[1], NULL, 10);
        if (index > BOOK_MAX || index < 0) {
            fprintf(stderr, "Valid values for bookmark indices are: 0 - %d\n",
                    BOOK_MAX - 1);
            return;
        }
        else if (bookmarks[index][0] == '\0') {
            fprintf(stderr, "No such bookmark exists.\n");
            return;
        }

        cd(bookmarks[index]);
    }
}
