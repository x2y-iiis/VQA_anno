#include <stdint.h>
#include <stddef.h>

/* All functions run through PyDLL with the GIL held. No allocation, waiting,
 * Python callbacks or GIL release. Python retains every node until removal. */
struct node { struct node *previous, *next; uint64_t status, identifier; };
struct gate {
    struct node *head, *tail;
    uint64_t maximum, ecot_limit, active, peak, grants, queued;
};

int vqa_gate_abi(void) { return 1; }
size_t vqa_gate_size(void) { return sizeof(struct gate); }
size_t vqa_node_size(void) { return sizeof(struct node); }

static void unlink_node(struct gate *gate, struct node *node) {
    if (node->previous) node->previous->next = node->next;
    else gate->head = node->next;
    if (node->next) node->next->previous = node->previous;
    else gate->tail = node->previous;
    node->previous = node->next = NULL;
    gate->queued--;
}

static uint64_t grant_one(struct gate *gate) {
    uint64_t limit = gate->ecot_limit && gate->ecot_limit < gate->maximum
        ? gate->ecot_limit : gate->maximum;
    if (!gate->head || gate->active >= limit) return 0;
    struct node *node = gate->head;
    unlink_node(gate, node);
    node->status = 2;
    gate->active++;
    gate->grants++;
    if (gate->active > gate->peak) gate->peak = gate->active;
    return node->identifier;
}

uint64_t vqa_gate_enter(struct gate *gate, struct node *node) {
    node->status = 1;
    node->previous = gate->tail;
    node->next = NULL;
    if (gate->tail) gate->tail->next = node;
    else gate->head = node;
    gate->tail = node;
    gate->queued++;
    return grant_one(gate);
}

uint64_t vqa_gate_leave(struct gate *gate, struct node *node) {
    if (node->status == 1) unlink_node(gate, node);
    else if (node->status == 2) gate->active--;
    node->status = 3;
    return grant_one(gate);
}

size_t vqa_gate_limit(struct gate *gate, uint64_t maximum, uint64_t ecot_limit,
                      uint64_t *wakeups) {
    gate->maximum = maximum;
    gate->ecot_limit = ecot_limit;
    size_t count = 0;
    uint64_t node;
    while ((node = grant_one(gate))) wakeups[count++] = node;
    return count;
}

void vqa_gate_copy(const struct gate *source, struct gate *target) { *target = *source; }
