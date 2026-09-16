#!/usr/bin/env python3
"""SAM 3 mask inference and deterministic nearest-mask point snapping."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import threading

import numpy as np


def nearest_mask_point(mask: np.ndarray, point_xy: list[float]) -> tuple[list[int], float]:
    """Return the mask pixel nearest to a pixel-space point."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError('sam3_mask_empty')
    point = np.asarray([float(point_xy[1]), float(point_xy[0])], dtype=np.float64)
    pixels = np.argwhere(binary)
    distances = np.sum((pixels - point[None, :]) ** 2, axis=1)
    index = int(np.argmin(distances))
    y, x = pixels[index]
    return [int(x), int(y)], float(distances[index] ** 0.5)


def choose_nearest_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    point_xy: list[float],
) -> tuple[np.ndarray, int, list[int], float, float]:
    """Select the SAM 3 instance whose mask is nearest to the supplied point."""
    candidates = np.asarray(masks)
    while candidates.ndim > 3 and candidates.shape[1] == 1:
        candidates = candidates[:, 0]
    if candidates.ndim == 2:
        candidates = candidates[None, ...]
    if candidates.ndim != 3 or not len(candidates):
        raise ValueError(f'sam3_mask_shape_invalid:{candidates.shape}')
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    ranked = []
    for index, mask in enumerate(candidates):
        if not np.asarray(mask, dtype=bool).any():
            continue
        snapped, distance = nearest_mask_point(mask, point_xy)
        score = float(values[index]) if index < len(values) else float('-inf')
        ranked.append((distance, -score, index, snapped, score))
    if not ranked:
        raise ValueError('sam3_all_candidate_masks_empty')
    distance, _, index, snapped, score = min(ranked)
    return np.asarray(candidates[index], dtype=bool), index, snapped, distance, score


def norm1000_to_pixel(point: list[float], width: int, height: int) -> list[float]:
    return [
        float(point[0]) / 1000.0 * max(1, width - 1),
        float(point[1]) / 1000.0 * max(1, height - 1),
    ]


def pixel_to_norm1000(point: list[float], width: int, height: int) -> list[int]:
    return [
        int(round(float(point[0]) / max(1, width - 1) * 1000)),
        int(round(float(point[1]) / max(1, height - 1) * 1000)),
    ]


class Sam3Snapper:
    """Lazily load official SAM 3 and snap points to text-prompted instance masks."""

    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        device: str = 'cuda',
        confidence_threshold: float = 0.25,
    ):
        self.repo = repo
        self.checkpoint = checkpoint
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._processor = None
        self._model = None
        self._torch = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._processor is not None:
            return
        if not self.repo.is_dir():
            raise RuntimeError(f'sam3_repo_missing:{self.repo}')
        if not self.checkpoint.is_file():
            raise RuntimeError(
                f'sam3_checkpoint_missing:{self.checkpoint}:request access to '
                'https://huggingface.co/facebook/sam3 and provide --sam3-checkpoint'
            )
        sys.path.insert(0, str(self.repo))
        try:
            import torch
            from PIL import Image
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as error:
            raise RuntimeError(
                f'sam3_import_failed:{error}:install the official SAM 3 package '
                f'from {self.repo}'
            ) from error
        model = build_sam3_image_model(
            checkpoint_path=str(self.checkpoint),
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
            enable_inst_interactivity=True,
        )
        self._torch = torch
        self._image_class = Image
        self._model = model
        self._processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

    def snap(
        self,
        image_rgb: np.ndarray,
        text_prompt: str,
        point_xy_1000: list[float],
        fallback_prompts: list[str] | None = None,
    ) -> dict:
        """Snap one normalized point to the nearest matching SAM 3 instance mask."""
        with self._lock:
            self._load()
            height, width = image_rgb.shape[:2]
            image = self._image_class.fromarray(
                np.asarray(image_rgb, dtype=np.uint8), mode='RGB',
            )
            with self._torch.inference_mode(), self._torch.autocast(
                device_type='cuda', dtype=self._torch.bfloat16,
                enabled=self.device.startswith('cuda'),
            ):
                state = self._processor.set_image(image)
            original_pixel = norm1000_to_pixel(point_xy_1000, width, height)
            prompts = []
            for prompt in [text_prompt, *(fallback_prompts or [])]:
                normalized = str(prompt).strip()
                if normalized and normalized.lower() not in {
                    value.lower() for value in prompts
                }:
                    prompts.append(normalized)
            failures = []
            for prompt in prompts:
                with self._torch.inference_mode(), self._torch.autocast(
                    device_type='cuda', dtype=self._torch.bfloat16,
                    enabled=self.device.startswith('cuda'),
                ):
                    output = self._processor.set_text_prompt(
                        state=state, prompt=prompt,
                    )
                masks = output['masks'].detach().float().cpu().numpy()
                scores = output['scores'].detach().float().cpu().numpy()
                try:
                    mask, index, snapped_pixel, distance, score = choose_nearest_mask(
                        masks, scores, original_pixel,
                    )
                except ValueError as error:
                    failures.append({'text_prompt': prompt, 'error': str(error)})
                    continue
                return {
                    'requested_text_prompt': text_prompt,
                    'text_prompt': prompt,
                    'mask_prompt_mode': 'text',
                    'prompt_attempts': [
                        *failures, {'text_prompt': prompt, 'status': 'selected'},
                    ],
                    'input_xy_1000': [int(round(value)) for value in point_xy_1000],
                    'snapped_xy_1000': pixel_to_norm1000(snapped_pixel, width, height),
                    'input_pixel_xy': [round(value, 3) for value in original_pixel],
                    'snapped_pixel_xy': snapped_pixel,
                    'distance_pixels': round(distance, 6),
                    'mask_index': index,
                    'mask_score': round(score, 8),
                    'mask_area_pixels': int(mask.sum()),
                    'image_size': [width, height],
                }
            with self._torch.inference_mode(), self._torch.autocast(
                device_type='cuda', dtype=self._torch.bfloat16,
                enabled=self.device.startswith('cuda'),
            ):
                masks, scores, _ = self._model.predict_inst(
                    state,
                    point_coords=np.asarray([original_pixel], dtype=np.float32),
                    point_labels=np.asarray([1], dtype=np.int32),
                    multimask_output=True,
                )
            mask, index, snapped_pixel, distance, score = choose_nearest_mask(
                masks, scores, original_pixel,
            )
            return {
                'requested_text_prompt': text_prompt,
                'text_prompt': None,
                'mask_prompt_mode': 'positive_point_fallback',
                'prompt_attempts': [
                    *failures,
                    {
                        'point_prompt_pixel_xy': [round(value, 3) for value in original_pixel],
                        'status': 'selected',
                    },
                ],
                'input_xy_1000': [int(round(value)) for value in point_xy_1000],
                'snapped_xy_1000': pixel_to_norm1000(snapped_pixel, width, height),
                'input_pixel_xy': [round(value, 3) for value in original_pixel],
                'snapped_pixel_xy': snapped_pixel,
                'distance_pixels': round(distance, 6),
                'mask_index': index,
                'mask_score': round(score, 8),
                'mask_area_pixels': int(mask.sum()),
                'image_size': [width, height],
            }


def map_crop_to_full(
    point_xy_1000: list[float],
    crop_box_pixel_xyxy: list[int],
    full_width: int,
    full_height: int,
) -> list[int]:
    x1, y1, x2, y2 = crop_box_pixel_xyxy
    crop_x = float(point_xy_1000[0]) / 1000.0
    crop_y = float(point_xy_1000[1]) / 1000.0
    full_x = x1 + crop_x * max(1, x2 - x1 - 1)
    full_y = y1 + crop_y * max(1, y2 - y1 - 1)
    return pixel_to_norm1000([full_x, full_y], full_width, full_height)


def snap_pair_twice(
    snapper: Sam3Snapper,
    crop_rgb: np.ndarray,
    full_rgb: np.ndarray,
    crop_box_pixel_xyxy: list[int],
    vlm_pair: dict,
    hand_prompt: str,
    object_prompt: str,
    hand_fallback_prompts: list[str] | None = None,
    object_fallback_prompts: list[str] | None = None,
) -> dict:
    """Snap in crop space, map to full space, then snap on full-image masks."""
    crop_hand = snapper.snap(
        crop_rgb, hand_prompt, vlm_pair['h_xy_1000'], hand_fallback_prompts,
    )
    crop_object = snapper.snap(
        crop_rgb, object_prompt, vlm_pair['o_xy_1000'], object_fallback_prompts,
    )
    full_height, full_width = full_rgb.shape[:2]
    mapped_hand = map_crop_to_full(
        crop_hand['snapped_xy_1000'], crop_box_pixel_xyxy, full_width, full_height,
    )
    mapped_object = map_crop_to_full(
        crop_object['snapped_xy_1000'], crop_box_pixel_xyxy, full_width, full_height,
    )
    full_hand = snapper.snap(
        full_rgb, hand_prompt, mapped_hand, hand_fallback_prompts,
    )
    full_object = snapper.snap(
        full_rgb, object_prompt, mapped_object, object_fallback_prompts,
    )
    return {
        'pair_index': int(vlm_pair.get('pair_index', 0)),
        'vlm_crop': {
            'h_xy_1000': [int(round(value)) for value in vlm_pair['h_xy_1000']],
            'o_xy_1000': [int(round(value)) for value in vlm_pair['o_xy_1000']],
        },
        'sam3_crop': {
            'h_xy_1000': crop_hand['snapped_xy_1000'],
            'o_xy_1000': crop_object['snapped_xy_1000'],
            'hand_mask': crop_hand,
            'object_mask': crop_object,
        },
        'mapped_full_before_sam3': {
            'h_xy_1000': mapped_hand,
            'o_xy_1000': mapped_object,
        },
        'sam3_full': {
            'h_xy_1000': full_hand['snapped_xy_1000'],
            'o_xy_1000': full_object['snapped_xy_1000'],
            'hand_mask': full_hand,
            'object_mask': full_object,
        },
        'h_xy_1000': full_hand['snapped_xy_1000'],
        'o_xy_1000': full_object['snapped_xy_1000'],
    }
