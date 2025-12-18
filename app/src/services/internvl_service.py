from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from app.src.config import Settings, get_settings
from app.src.state.session_store import SessionStore
from app.src.utils.image_ops import DetectedObject
from app.src.utils.validation import ValidationError


class InternVLService:
    """Thin wrapper that exposes higher-level tasks on top of InternVL3.5."""

    def __init__(self, settings: Settings | None = None, session_store: SessionStore | None = None) -> None:
        self.settings = settings or get_settings()
        self.session_store = session_store or SessionStore()
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._processor: Optional[AutoProcessor] = None
        self._device: Optional[torch.device] = None
        self._lock = threading.Lock()

        # Pre-load model immediately to avoid timeout on first request
        print("Initializing InternVLService: Loading model weights... (this may take a minute)")
        self._load_model()
        print("InternVLService: Model loaded and ready.")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def generate_caption(
        self,
        image: Optional[Image.Image],
        question: Optional[str],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Produce a caption or answer a question about the image."""

        image = self._resolve_image(session_id, image)
        if image is None:
            raise ValidationError("Сначала загрузите изображение.")

        session_id = session_id or self.session_store.create_session(image)
        if image is not None:
            self.session_store.update_image(session_id, image)

        prompt = self._build_vqa_prompt(question)
        answer = self._generate(image, prompt)

        user_turn = question.strip() if question else "Опиши изображение"
        self.session_store.append_exchange(session_id, user_turn, answer)

        return {
            "session_id": session_id,
            "answer": answer,
            "history": self.session_store.get_history(session_id),
        }

    def find_objects(self, image: Image.Image, text_query: str) -> List[DetectedObject]:
        image = image.convert("RGB")
        prompt = self._build_grounding_prompt(text_query)
        # We need special tokens to parse <box> and <ref> tags if they are generated
        raw = self._generate(image, prompt, skip_special_tokens=False)
        return self._parse_grounding_output(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_image(
        self, session_id: Optional[str], provided: Optional[Image.Image]
    ) -> Optional[Image.Image]:
        if provided is not None:
            return provided.convert("RGB")
        if session_id and self.session_store.has_session(session_id):
            return self.session_store.get_image(session_id)
        return None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            dtype = torch.float32
            if self.settings.resolved_device == "cuda":
                dtype = torch.bfloat16
            elif self.settings.resolved_device == "mps":
                dtype = torch.float16

            self._device = torch.device(self.settings.resolved_device)

            print(f"DEBUG: Loading model from {self.settings.model_name}")
            print(f"DEBUG: Device: {self._device}, Dtype: {dtype}")
            
            # Explicitly checking file existence
            import os
            model_path = str(self.settings.model_name)
            
            # Monkey-patch Qwen2Tokenizer classes to add attributes required by InternVLProcessor
            # This fixes 'AttributeError: Qwen2TokenizerFast has no attribute start_image_token[_id]' and friends.
            try:
                # Try importing from the specific modules to be sure we get the actual classes
                from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
                from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
                targets = [Qwen2Tokenizer, Qwen2TokenizerFast]

                for cls in targets:
                    # String tokens (used to build prompts / templates)
                    if not hasattr(cls, "start_image_token"):
                        setattr(cls, "start_image_token", "<img>")
                    if not hasattr(cls, "end_image_token"):
                        setattr(cls, "end_image_token", "</img>")
                    if not hasattr(cls, "img_context_token"):
                        setattr(cls, "img_context_token", "<IMG_CONTEXT>")
                    if not hasattr(cls, "context_image_token"):
                        setattr(cls, "context_image_token", "<IMG_CONTEXT>")
                    if not hasattr(cls, "video_token"):
                        setattr(cls, "video_token", "<|video_pad|>")

                print("DEBUG: Successfully patched Qwen2Tokenizer classes (string attributes).")
            except (ImportError, AttributeError) as e:
                print(f"DEBUG: Could not patch Qwen2Tokenizer classes: {e}. If using Qwen2-based model, this might fail.")

            if os.path.exists(model_path):
                print(f"DEBUG: Found local model path: {model_path}")
                # print(f"DEBUG: Files: {os.listdir(model_path)}")
            else:
                print(f"DEBUG: Model path '{model_path}' not found locally, relying on HF Hub logic.")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.settings.model_name,
                trust_remote_code=True,
                cache_dir=self.settings.hf_cache_dir,
                use_fast=False,
            )
            print("DEBUG: Tokenizer loaded.")
            
            # Ensure the tokenizer instance has the string attributes as well (redundancy)
            if not hasattr(self._tokenizer, "start_image_token"):
                self._tokenizer.start_image_token = "<img>"
            if not hasattr(self._tokenizer, "end_image_token"):
                self._tokenizer.end_image_token = "</img>"
            if not hasattr(self._tokenizer, "img_context_token"):
                self._tokenizer.img_context_token = "<IMG_CONTEXT>"
            if not hasattr(self._tokenizer, "context_image_token"):
                self._tokenizer.context_image_token = "<IMG_CONTEXT>"
            if not hasattr(self._tokenizer, "video_token"):
                self._tokenizer.video_token = "<|video_pad|>"

            # Populate *_token_id on the tokenizer instance and on Qwen2* classes,
            # so older InternVLProcessor implementations that access tokenizer.start_image_token_id
            # directly will work inside Docker.
            try:
                start_img_id = self._tokenizer.convert_tokens_to_ids("<img>")
                end_img_id = self._tokenizer.convert_tokens_to_ids("</img>")
                img_ctx_id = self._tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
                video_id = self._tokenizer.convert_tokens_to_ids("<|video_pad|>")
            except Exception as e:  # very defensive; should not happen for this model
                print(f"DEBUG: Failed to convert special tokens to ids: {e}")
                start_img_id = getattr(self._tokenizer, "start_image_token_id", None)
                end_img_id = getattr(self._tokenizer, "end_image_token_id", None)
                img_ctx_id = getattr(self._tokenizer, "img_context_token_id", None)
                video_id = getattr(self._tokenizer, "video_token_id", None)

            # Instance-level IDs
            if not hasattr(self._tokenizer, "start_image_token_id") or self._tokenizer.start_image_token_id is None:
                self._tokenizer.start_image_token_id = start_img_id
            if not hasattr(self._tokenizer, "end_image_token_id") or self._tokenizer.end_image_token_id is None:
                self._tokenizer.end_image_token_id = end_img_id
            if not hasattr(self._tokenizer, "img_context_token_id") or self._tokenizer.img_context_token_id is None:
                self._tokenizer.img_context_token_id = img_ctx_id
            if not hasattr(self._tokenizer, "context_image_token_id") or self._tokenizer.context_image_token_id is None:
                self._tokenizer.context_image_token_id = img_ctx_id
            if not hasattr(self._tokenizer, "video_token_id") or self._tokenizer.video_token_id is None:
                self._tokenizer.video_token_id = video_id

            # Also patch the Qwen2 classes with the computed IDs so that any tokenizer instances
            # created inside AutoProcessor.from_pretrained also have them.
            try:
                from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
                from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
                for cls in (Qwen2Tokenizer, Qwen2TokenizerFast):
                    if not hasattr(cls, "start_image_token_id"):
                        setattr(cls, "start_image_token_id", start_img_id)
                    if not hasattr(cls, "end_image_token_id"):
                        setattr(cls, "end_image_token_id", end_img_id)
                    if not hasattr(cls, "img_context_token_id"):
                        setattr(cls, "img_context_token_id", img_ctx_id)
                    if not hasattr(cls, "context_image_token_id"):
                        setattr(cls, "context_image_token_id", img_ctx_id)
                    if not hasattr(cls, "video_token_id"):
                        setattr(cls, "video_token_id", video_id)
                print("DEBUG: Patched Qwen2Tokenizer classes with token IDs.")
            except Exception as e:
                print(f"DEBUG: Failed to patch Qwen2Tokenizer classes with IDs: {e}")


            self._processor = AutoProcessor.from_pretrained(
                self.settings.model_name,
                trust_remote_code=True,
                cache_dir=self.settings.hf_cache_dir,
            )
            print("DEBUG: Processor loaded.")

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
                "cache_dir": self.settings.hf_cache_dir,
                "low_cpu_mem_usage": True,
            }
            if self._device.type == "cuda":
                model_kwargs["device_map"] = "auto"

            print("DEBUG: Loading model weights (this is the slow part)...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_name,
                **model_kwargs,
            )
            print("DEBUG: Model weights loaded object.")

            # Fix for missing img_context_token_id in InternVL 3.5
            if getattr(self._model, "img_context_token_id", None) is None:
                 img_context_id = self._tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
                 if img_context_id is not None:
                     self._model.img_context_token_id = img_context_id

            if self._device.type != "cuda":
                self._model.to(self._device)
            self._model.eval()

    def _build_vqa_prompt(self, question: Optional[str]) -> str:
        if question:
            return (
                "<IMG_CONTEXT>\nYou are an expert visual assistant. Answer the question using only the "
                "information visible in the image.\nQuestion: "
                + question.strip()
            )
        return "<IMG_CONTEXT>\nPlease describe the image in 2-3 concise sentences."

    def _build_grounding_prompt(self, query: str) -> str:
        # Strong instruction with specific output format for 1B model
        return (
            f"<|im_start|>user\n"
            f"<IMG_CONTEXT>\nFind <ref>{query.strip()}</ref> in the image. "
            f"Output the bounding box coordinates in the format [[x1, y1, x2, y2]].<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _generate(self, image: Image.Image, prompt: str, skip_special_tokens: bool = True) -> str:
        self._load_model()
        assert self._processor and self._model and self._tokenizer and self._device

        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        
        # Determine target dtype from model parameters
        target_dtype = self._model.dtype

        inputs = {
            k: v.to(self._device) if hasattr(v, "to") else v 
            for k, v in inputs.items()
        }

        # Explicitly cast pixel_values to model dtype (e.g. bfloat16)
        if "pixel_values" in inputs:
             inputs["pixel_values"] = inputs["pixel_values"].to(target_dtype)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.max_tokens,
                do_sample=False,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)[0]
        return self._post_process(decoded, prompt)

    @staticmethod
    def _post_process(text: str, prompt: str) -> str:
        cleaned = text.strip()
        
        # Handle cases where <IMG_CONTEXT> token might be missing in decoded text
        prompt_clean = prompt.replace("<IMG_CONTEXT>", "").strip()
        
        if cleaned.startswith(prompt):
            cleaned = cleaned[len(prompt) :].strip()
        elif cleaned.startswith(prompt_clean):
            cleaned = cleaned[len(prompt_clean) :].strip()
            
        return cleaned

    def _parse_grounding_output(self, text: str) -> List[DetectedObject]:
        # InternVL outputs: <ref>object</ref><box>[[x1, y1, x2, y2]]</box>
        
        payload = []
        import re
        
        # 1. Try to find <ref>label</ref><box>coords</box> pairs
        # This handles cases where the model lists multiple objects with labels
        ref_box_pattern = r"<ref>(.*?)</ref>\s*<box>(.*?)</box>"
        matches = re.findall(ref_box_pattern, text)
        
        if matches:
            for label, box_str in matches:
                label = label.strip()
                # Parse inner numbers
                nums = re.findall(r"([\d\.]+)", box_str)
                for i in range(0, len(nums), 4):
                    if i + 3 >= len(nums): break
                    try:
                        coords = [float(nums[j]) for j in range(i, i+4)]
                        if any(c > 1.0 for c in coords):
                            coords = [c / 1000.0 for c in coords]
                        payload.append({"box": coords, "label": label, "confidence": 1.0})
                    except ValueError:
                        continue
        
        # 2. Fallback: Try to find standalone <box>...</box> content if no ref pairs found
        if not payload:
            box_pattern = r"<box>(.*?)</box>"
            box_matches = re.findall(box_pattern, text)
            
            # If no tags, look for raw list pattern [[...]]
            if not box_matches:
                 box_matches = re.findall(r"\[\[.*?\]\]", text)
            
            for box_str in box_matches:
                nums = re.findall(r"([\d\.]+)", box_str)
                for i in range(0, len(nums), 4):
                    if i + 3 >= len(nums): break
                    try:
                        coords = [float(nums[j]) for j in range(i, i+4)]
                        if any(c > 1.0 for c in coords):
                            coords = [c / 1000.0 for c in coords]
                        payload.append({"box": coords, "label": "object", "confidence": 1.0})
                    except ValueError:
                        continue

        # 3. Last resort fallback
        if not payload:
             matches = re.findall(r"\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]", text)
             for match in matches:
                 try:
                     coords = [float(x) for x in match]
                     if any(c > 1.0 for c in coords):
                         coords = [c / 1000.0 for c in coords]
                     payload.append({"box": coords, "label": "object", "confidence": 1.0})
                 except ValueError:
                     continue

        if not payload:
            print(f"DEBUG: Failed to parse grounding output. Raw text: {text}")
            raise ValidationError(f"Не удалось найти объекты. Ответ модели: {text[:100]}...")

        detections: List[DetectedObject] = []
        for item in payload:
            try:
                label = str(item.get("label", "object")).strip() or "object"
                confidence = float(item.get("confidence", 0.0))
                xmin, ymin, xmax, ymax = map(float, item.get("box", [0, 0, 1, 1]))
            except (TypeError, ValueError) as exc:
                raise ValidationError("Получен неожиданный формат боксов.") from exc

            if confidence < self.settings.object_score_threshold:
                continue
            xmin = max(0.0, min(1.0, xmin))
            ymin = max(0.0, min(1.0, ymin))
            xmax = max(xmin, min(1.0, xmax))
            ymax = max(ymin, min(1.0, ymax))
            detections.append(DetectedObject(label=label, confidence=confidence, box=(xmin, ymin, xmax, ymax)))

        if not detections:
            raise ValidationError("Объекты по данному описанию не найдены.")
        return detections

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        fenced = re.search(r"```json(.*?)```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        bracket_start = text.find("[")
        bracket_end = text.rfind("]")
        if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
            return text[bracket_start : bracket_end + 1].strip()
        return None
