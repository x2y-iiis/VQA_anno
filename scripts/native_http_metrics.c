#include <stdint.h>
#include <stddef.h>

#define VQA_RECENT_CALLS 2048

/* Called exclusively through ctypes.PyDLL with the CPython GIL held.
 * No I/O, waiting, callbacks, allocation, or GIL release is allowed here.
 * Each update/snapshot is coherent without an additional contended mutex. */
struct vqa_http_metrics {
    uint64_t active, peak, started, finished, failed;
    double completed_seconds;
    uint64_t recent_count;
    double recent[VQA_RECENT_CALLS];
};

int vqa_http_metrics_abi(void) { return 1; }
size_t vqa_http_metrics_size(void) { return sizeof(struct vqa_http_metrics); }

void vqa_http_metrics_start(struct vqa_http_metrics *state) {
    state->started++;
    state->active++;
    if (state->active > state->peak) state->peak = state->active;
}

void vqa_http_metrics_finish(struct vqa_http_metrics *state, int failed, double elapsed) {
    state->active--;
    state->recent[state->finished % VQA_RECENT_CALLS] = elapsed;
    state->finished++;
    state->failed += !!failed;
    state->completed_seconds += elapsed;
    if (state->recent_count < VQA_RECENT_CALLS) state->recent_count++;
}

void vqa_http_metrics_copy(const struct vqa_http_metrics *source, struct vqa_http_metrics *target) {
    *target = *source;
}
