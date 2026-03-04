"""
Utility Functions for SmolLM Training

Includes:
- Logging setup (singleton pattern for unified log file)
- Pause/resume handler
- Device management utilities
- Generation functions
"""

import os
import signal
import time
import logging
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn


# ============================================================================
# LOGGING SETUP (Singleton Pattern - All Modules Use Same Logger)
# ============================================================================

_logger = None  # Global logger instance

def setup_logging(log_file="training_logs.txt"):
    """
    Setup file logging with singleton pattern.
    All modules should call this ONCE at startup, then use get_logger().

    Returns:
        logging.Logger: Configured logger instance
    """
    global _logger
    if _logger is not None:
        return _logger

    # Clear any existing handlers to prevent duplicates
    logging.root.handlers = []

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    _logger = logging.getLogger()
    return _logger


def get_logger():
    """
    Get the global logger instance.
    Call this in any module that needs to log.
    """
    global _logger
    if _logger is None:
        raise RuntimeError("Logger not initialized. Call setup_logging() first.")
    return _logger


# ============================================================================
# PAUSE/RESUME HANDLER
# ============================================================================

class PauseHandler:
    """Handles pause/resume via signals and file flag"""
    def __init__(self, pause_flag_path: str = "checkpoints/.pause"):
        self.pause_flag_path = Path(pause_flag_path)
        self.pause_flag_path.parent.mkdir(parents=True, exist_ok=True)
        self.paused = False
        self.in_pause_wait = False
        self.should_save = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if self.in_pause_wait:
            print(f"\n[EXIT] Received signal {signum} while paused. Exiting...")
            raise KeyboardInterrupt

        print(f"\n[PAUSE] Received signal {signum}. Pausing after current step...")
        self.paused = True
        self.should_save = True

    def check_pause_flag(self) -> bool:
        return self.pause_flag_path.exists()

    def clear_pause_flag(self):
        if self.pause_flag_path.exists():
            self.pause_flag_path.unlink()

    def should_pause(self) -> bool:
        return self.paused or self.check_pause_flag()

    def resume(self):
        self.paused = False
        self.clear_pause_flag()


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def sync_device(device: torch.device):
    """Synchronize device for accurate timing"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def set_runtime_optimizations(device: torch.device):
    """Set runtime optimizations for MPS (Mac M1)"""
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # MPS doesn't need cudnn settings (that's CUDA-specific)


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

@torch.no_grad()
def sample_generate_single_fast(
    model: nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
) -> torch.Tensor:
    """
    Single-prompt generation with preallocated token buffer (no torch.cat per token).
    Returns: (T_out,) LongTensor on same device.
    Assumes model(input_ids) -> logits of shape (B, T, V).
    """
    device = prompt_ids.device
    vocab_size = model.vocab_size
    max_valid_id = vocab_size - 1

    # Truncate prompt to fit (leave room for at least 1 generated token)
    prompt_len = min(int(prompt_ids.size(0)), max_seq_len - 1)
    prompt_ids = prompt_ids[:prompt_len]

    # Preallocate full buffer and copy prompt in
    buf = torch.empty((1, max_seq_len), dtype=torch.long, device=device)
    buf[:, :prompt_len] = prompt_ids.unsqueeze(0)

    cur_len = prompt_len
    max_total_len = min(max_seq_len, prompt_len + max_new_tokens)

    for _ in range(max_new_tokens):
        if cur_len >= max_total_len:
            break

        # Forward only on the filled prefix
        outputs = model(buf[:, :cur_len])
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        next_logits = logits[:, -1, :]

        # Defensive: truncate if model outputs extra dims
        if next_logits.size(-1) > vocab_size:
            next_logits = next_logits[:, :vocab_size]

        if temperature <= 0:
            next_id = torch.argmax(next_logits, dim=-1)
        else:
            next_logits = next_logits / float(temperature)
            probs = torch.softmax(next_logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf <= top_p
                mask[:, 0] = True

                # Zero out tokens outside nucleus and renormalize
                filtered = sorted_probs * mask.float()
                filtered = filtered / filtered.sum(dim=-1, keepdim=True)

                # Sample in sorted space, then map back to vocab ids
                sampled_in_sorted = torch.multinomial(filtered, num_samples=1)
                next_id = sorted_idx.gather(-1, sampled_in_sorted).squeeze(-1)
            else:
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Defensive clamp
        next_id = torch.clamp(next_id, min=0, max=max_valid_id)

        # Write next token into buffer
        buf[0, cur_len] = next_id[0]
        cur_len += 1

    return buf[0, :cur_len]


def sample_generate_batch(
    model: nn.Module,
    tokenizer,
    prompt_ids_list: list,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
) -> list:
    """
    Batched generation for multiple prompts.
    prompt_ids_list: list of (T,) LongTensors on device
    max_seq_len: maximum sequence length (model's max_seq_len)
    returns: list of generated tensors
    """
    if not prompt_ids_list:
        return []

    device = prompt_ids_list[0].device
    batch_size = len(prompt_ids_list)

    max_prompt_len = max(p.size(0) for p in prompt_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    if max_prompt_len >= max_seq_len:
        max_prompt_len = max_seq_len - 1
        prompt_ids_list = [p[:max_prompt_len] for p in prompt_ids_list]
        max_prompt_len = max(p.size(0) for p in prompt_ids_list) if prompt_ids_list else 0

    max_prompt_len = min(max_prompt_len, max_seq_len - 1)

    batch = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids_list):
        batch[i, :p.size(0)] = p

    # Generate tokens
    vocab_size = model.vocab_size
    for step in range(max_new_tokens):
        # Check if adding another token would exceed max_seq_len
        current_seq_len = batch.size(1)
        if current_seq_len >= max_seq_len:
            break

        # Safety check: truncate batch if it somehow exceeds max_seq_len
        if current_seq_len > max_seq_len:
            batch = batch[:, :max_seq_len]
            break

        # NO autocast in generation (matching deepscreen exactly)
        outputs = model(batch)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        next_logits = logits[:, -1]

        # Clamp logits to valid vocab range
        max_valid_id = vocab_size - 1
        if next_logits.shape[-1] > vocab_size:
            next_logits = next_logits[:, :vocab_size]

        if temperature <= 0:
            next_ids = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / float(temperature)
            probs = torch.softmax(next_logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf <= top_p
                mask[:, 0] = True
                probs = probs.scatter(-1, sorted_idx, sorted_probs * mask.float())
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_ids = torch.multinomial(probs, num_samples=1)

        # Clamp generated IDs to valid range
        next_ids = torch.clamp(next_ids, min=0, max=max_valid_id)

        # Check if adding this token would exceed max_seq_len
        if current_seq_len + 1 > max_seq_len:
            break

        batch = torch.cat([batch, next_ids], dim=1)

    return [batch[i] for i in range(batch_size)]


# ============================================================================
# SCHEDULE FUNCTIONS
# ============================================================================

def tau_schedule(step: int) -> float:
    """Temperature schedule for training"""
    if step <= 1000:
        return 2.0 - (step / 1000.0)
    return 1.0
