#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>

static int arg = -1;
module_param(arg, int, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

int procinfo_init(void)
{
    struct task_struct *task, *sibling;
    struct list_head *list;

    char      *name, *state_desc, *prio_desc, *static_prio_desc, *policy_desc;
    pid_t     pid, real_parent_pid, parent_pid;
    long      state;
    int       prio, static_prio, policy;
    cputime_t time_slice;

    printk(KERN_INFO "Loading ProcessInfo Module\n");

    for_each_process(task) {
        if ((arg == -1 || arg == task->pid)) {

            name = task->comm;
            pid = task->pid;
            state = task->state;
            state_desc = state > 0 ? "Stopped"
                : state < 0 ? "Unrunnable"
                : "Runnable";
            prio = task->prio;
            prio_desc = prio < 100 ? "Realtime" : "Normal";
            static_prio = task->static_prio;
            static_prio_desc = static_prio < 100 ? "Realtime" : "Normal";
            real_parent_pid = task->real_parent->pid;
            parent_pid = task->parent->pid;
            time_slice = task->utime + task->stime;
            policy = task->policy;
            switch (policy) {
            case 0:
                policy_desc = "SCHED_NORMAL";
                break;
            case 1:
                policy_desc = "SCHED_FIFO";
                break;
            case 2:
                policy_desc = "SCHED_RR";
                break;
            case 3:
                policy_desc = "SCHED_BATCH";
                break;
            case 5:
                policy_desc = "SCHED_IDLE";
                break;
            case 6:
                policy_desc = "SCHED_DEADLINE";
                break;
            default:
                policy_desc = "OTHER";
                break;
            }

            printk("Executable Name : %s\n", name);
            printk("Process ID      : %ld\n", (long)pid);
            printk("State           : %ld (%s)\n", (long)state, state_desc);
            printk("Priority        : %d (%s)\n", prio, prio_desc);
            printk("Static Priority : %d (%s)\n", static_prio, static_prio_desc);
            printk("Real Parent PID : %ld\n", (long)real_parent_pid);
            printk("Parent PID      : %ld\n", (long)parent_pid);
            printk("Time Slice      : %d\n", (int)time_slice);
            printk("Policy          : %d (%s)\n", policy, policy_desc);

            list_for_each(list, &task->sibling) {
                sibling = list_entry(list, struct task_struct, sibling);
                if (sibling->pid != 0) {
                    printk("Sibling Executable Name : %s\n", sibling->comm);
                    printk("Sibling Process ID      : %ld\n", (long)sibling->pid);
                }
            }
        }
    }

    return 0;
}

void procinfo_exit(void)
{
    printk(KERN_INFO "Removing ProcessInfo Module\n");
}

module_init(procinfo_init);
module_exit(procinfo_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Process Info Module");
MODULE_AUTHOR("Emre");
