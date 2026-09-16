#include <errno.h>
#include <limits.h>
#include <linux/futex.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

/* A generation is signalled once and stays readable until its last waiter
 * drops the owning Python object. No Python waiter-by-waiter wakeup loop. */
int vqa_latch_abi(void) { return 1; }

int vqa_latch_wait(uint32_t *word, int timeout_ms) {
    if (__atomic_load_n(word, __ATOMIC_ACQUIRE)) return 1;
    struct timespec timeout = {
        .tv_sec = timeout_ms / 1000,
        .tv_nsec = (long)(timeout_ms % 1000) * 1000000L
    };
    long result = syscall(SYS_futex, word, FUTEX_WAIT_PRIVATE, 0,
                          &timeout, NULL, 0);
    if (result < 0 && errno != EAGAIN && errno != EINTR && errno != ETIMEDOUT)
        return -errno;
    return __atomic_load_n(word, __ATOMIC_ACQUIRE) ? 1 : 0;
}

int vqa_latch_notify(uint32_t *word) {
    __atomic_store_n(word, 1, __ATOMIC_RELEASE);
    long result = syscall(SYS_futex, word, FUTEX_WAKE_PRIVATE, INT_MAX,
                          NULL, NULL, 0);
    return result < 0 ? -errno : 0;
}
