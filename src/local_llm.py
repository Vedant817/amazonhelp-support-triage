"""Revision-pinned local instruction-model runtime; no hosted API or key required."""

from __future__ import annotations

from typing import Dict, List, Optional


QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


class LocalInstructionModel:
    def __init__(
        self,
        model_name: str = QWEN_MODEL_ID,
        revision: str = QWEN_REVISION,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        selected_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if selected_device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=False,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            torch_dtype=dtype,
            trust_remote_code=False,
        ).to(selected_device)
        self._model.generation_config.temperature = None
        self._model.generation_config.top_p = None
        self._model.generation_config.top_k = None
        self.device = selected_device

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 128) -> str:
        self._load()
        import torch

        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        message_batches: List[List[Dict[str, str]]],
        max_new_tokens: int = 128,
        batch_size: int = 4,
    ) -> List[str]:
        """Deterministic padded batch generation for CPU-efficient local evaluation."""
        self._load()
        import torch

        if not message_batches:
            return []
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        outputs: List[str] = []
        for start in range(0, len(message_batches), max(1, batch_size)):
            chunk = message_batches[start : start + max(1, batch_size)]
            prompts = [
                self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in chunk
            ]
            inputs = self._tokenizer(
                prompts, return_tensors="pt", padding=True
            ).to(self._model.device)
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            prompt_length = inputs["input_ids"].shape[1]
            outputs.extend(
                self._tokenizer.decode(
                    row[prompt_length:], skip_special_tokens=True
                ).strip()
                for row in generated
            )
        return outputs
