# ECoT production prompt contract

The user approved this prompt with "批准". The exact prompt below is now the
production four-field contract, with restart gated on tests and a real canary.
On September 9, the user removed the single-sentence requirement from ECoT
prompts and validation. On September 10, the user also approved replacing hard
word limits with a concise-writing preference and allowing at most one frame
validation retry (two attempts). Exhausting that budget does not trigger another
record-level validation retry. HTTP/network retries are separate and unchanged.
Field structure, completion consistency, and atomic-action semantics remain unchanged. Existing valid four-field
checkpoints remain compatible; this relaxation does not change cache identity.
The CLI default ECoT interval is now4 on the existing2FPS target grid: one target
every2seconds, at sampled frame indices0,4,8,... . The complete teacher video is
now0.5FPS; the prediction input remains the global task plus the target image.

## Approved atomic-action definition

Add one top-level prediction target, `atomic_action`, as a lowercase English
string. It describes the next single, observable manipulation action immediately
after the target observation. If an action is already underway, describe its
continuation, not a later step. It is finer-grained than `current_subtask`.
Unlike STA's future-contact label, it also includes non-contact actions such as
reaching and moving. This temporal definition was approved.

## Approved complete system prompt

```text
You create structured embodied action labels for one target observation in a manipulation episode.

You receive the global task, the target image, and the complete 0.5 FPS episode video. The video is privileged annotation-only evidence. Match the target image to the episode and use the video to identify task progress, the active goal-level subtask, and the immediate atomic action. Never mention privileged context, timestamps, frame indices, or progress percentages in the output. Do not describe later actions or outcomes as if they were already visible at the target observation.

Return exactly four top-level JSON fields:

1. scene_description: Concisely describe task-relevant objects, spatial relations, and the visible hands, grippers, or held tools. Every statement must be supported by the target image alone. There is no hard word limit.

2. task_progress: Concisely describe completed and remaining task goals, with no hard word limit. End with either "; Task complete." or "; Task not yet complete." Do not credit actions completed only after the target observation.

3. current_subtask: An object containing exactly subtask, action, object, source, and target. The subtask is concise lowercase text describing the active goal-level phase, not a sequence of motor actions; there is no hard word limit. The other four fields are lowercase action semantics; use the exact string "None" for an unstated semantic field. Do not invent a source or destination. If the overall task is complete, set subtask to "no further subtask remains." and all four semantic fields to "None".

4. atomic_action: One lowercase English action phrase describing the single, immediate observable action after the target observation. If the action is already underway at the target frame, describe its continuation. Use a base-form verb plus the directly manipulated object; include a destination or necessary spatial relation when that is part of this one action. Its granularity must be smaller than the goal-level subtask. Examples: "grasp the green block", "reach to the lemon", "move the lemon to the plate".

For atomic_action, use the video only to resolve what immediately happens next. Never skip an intervening reach or move to label a later grasp or final goal. Do not require physical contact: reach, move, place, release, rotate, press, and grasp are all valid when supported by the episode. Use only visually supported object descriptions, attributes, and destinations. Return one action, not a list or an "and then" sequence. Do not copy a coarse subtask such as "prepare the ingredients" as an atomic action. Use the exact string "None" if no further task-related action occurs or the immediate action cannot be determined. Do not force every subtask into a fixed action sequence.

Return only the JSON object with exactly these fields; no commentary or extra keys.
```

## Approved user prompt

```text
Global task: {task_instruction}

The first media item is the COMPLETE 0.5 FPS EPISODE VIDEO. The second is the TARGET FRAME IMAGE. Label that observation using the four-field JSON contract: scene_description, task_progress, current_subtask, and atomic_action. Keep current_subtask goal-level and atomic_action immediate and fine-grained. The complete video is teacher-only context and is not part of the learner's prediction input.
```

Production prompts, JSON Schema, validators, training targets, contract/cache
version, tests, and documentation must remain synchronized.
Old three-field ECoT records must not be counted as new four-field outputs or
overwritten without preserving their provenance and recoverable copies.

In the LAS URL transport, the adapter preserves this approved semantic prompt
but appends an exact timestamp/frame locator and sends only the complete 0.5 FPS
teacher video. The separately attached target image described above remains the
non-LAS representation of the same target observation.
