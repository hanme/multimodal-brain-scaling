from dataclasses import dataclass
from typing import Optional, Union, Any, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
# from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from transformers import AutoVideoProcessor, AutoModel



ImageLike = Union[
    Image.Image,          # PIL
    np.ndarray,           # HWC or CHW
    torch.Tensor,         # CHW or HWC
    str,                  # URL or path
    Path,                 # local path
]


def _to_pil(img: ImageLike) -> Union[Image.Image, str]:
    """
    Convert common image formats to PIL.Image.
    If img is a URL string, return it as-is (many HF multimodal processors accept URLs).
    """
    if isinstance(img, Image.Image):
        return img

    if isinstance(img, Path):
        return Image.open(img).convert("RGB")

    if isinstance(img, str):
        # If it's a local path, load it; otherwise treat as URL
        p = Path(img)
        if p.exists():
            return Image.open(p).convert("RGB")
        return img  # URL or remote identifier

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            # best-effort: assume [0,1] or [0,255]
            arr = np.clip(arr, 0, 1) if arr.max() <= 1.0 else np.clip(arr, 0, 255)
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    if torch.is_tensor(img):
        t = img.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW
            # TF.to_pil_image handles float [0,1] and uint8
            return TF.to_pil_image(t).convert("RGB")
        if t.ndim == 3 and t.shape[-1] in (1, 3):  # HWC
            return TF.to_pil_image(t.permute(2, 0, 1)).convert("RGB")
        raise ValueError(f"Unsupported tensor shape for image: {tuple(t.shape)}")

    raise TypeError(f"Unsupported image type: {type(img)}")


@dataclass
class HFChatImagePreprocessor:
    """
    Wraps a HF AutoProcessor to produce model-compatible inputs from a single image.

    Behavior:
      - If fixed_prompt is not None: adds that same text for every call.
      - If fixed_prompt is None: no text is added (text inputs are discarded).
    """
    processor: Any
    fixed_prompt: Optional[str] = None
    add_generation_prompt: bool = False  # whether to add the special generation prompt token (if supported by the processor/model)
    tokenize: bool = True
    return_dict: bool = True
    return_tensors: str = "pt"

    def __call__(
        self,
        image: ImageLike,
        image_resize: Optional[Union[int, tuple]] = 256,
        *,
        device: Optional[Union[str, torch.device]] = None,
        prompt: Optional[str] = None,
        extra_apply_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          image: a single image (PIL/numpy/tensor/path/URL).
          device: optional device to move outputs to.
          prompt: optional override. If provided, used instead of fixed_prompt.
                  If explicitly prompt=None, and fixed_prompt=None -> text is omitted.
          extra_apply_kwargs: forwarded to apply_chat_template (e.g., {"padding": True} if supported).

        Returns:
          A dict of tensors compatible with HF model.forward(...)
        """
        img_obj = _to_pil(image)
        if image_resize is not None:
            if isinstance(image_resize, int):
                image_resize = (image_resize, image_resize)
            img_obj = img_obj.resize(image_resize, resample=Image.BICUBIC)

        text = prompt if prompt is not None else self.fixed_prompt

        content = [{"type": "image", "image": img_obj}]
        if text is not None and str(text).strip() != "":
            content.append({"type": "text", "text": str(text)})

        conversation = [{"role": "user", "content": content}]

        kwargs = dict(
            tokenize=self.tokenize,
            add_generation_prompt=self.add_generation_prompt,
            return_dict=self.return_dict,
            return_tensors=self.return_tensors,
        )
        if extra_apply_kwargs:
            kwargs.update(extra_apply_kwargs)

        try:
            # most Qwen/VL examples: apply_chat_template(conversation, ...)
            inputs = self.processor.apply_chat_template(conversation, **kwargs)
        except TypeError:
            try:
                # some versions: apply_chat_template(conversation=..., ...)
                inputs = self.processor.apply_chat_template(conversation=conversation, **kwargs)
            except TypeError:
                # fallback: some accept messages=...
                inputs = self.processor.apply_chat_template(messages=conversation, **kwargs)


        if device is not None:
            inputs = inputs.to(device)

        return inputs

class HFVideoPreprocessor:
    """
    Similar to HFChatImagePreprocessor but for video inputs. Wraps a HF AutoVideoProcessor.
    For simplicity, this example assumes the video processor can handle raw video tensors or paths/URLs directly.
    """

    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(
        self,
        video: Union[torch.Tensor, str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        extra_apply_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          video: a video tensor (e.g., [T,C,H,W]) or a path/URL to a video file.
          device: optional device to move outputs to.
          extra_apply_kwargs: forwarded to the processor's __call__.

        Returns:
          A dict of tensors compatible with HF model.forward(...)
        """
        kwargs = dict(return_tensors="pt")
        if extra_apply_kwargs:
            kwargs.update(extra_apply_kwargs)

        inputs = self.processor(video, **kwargs)

        if device is not None:
            inputs = inputs.to(device)

        return inputs


def load_model_hf(model_id:str, **kwargs) -> nn.Module:
    fixed_prompt = kwargs.get("prompt")
    add_generation_prompt = kwargs.get("add_generation_prompt", False)

    try:
        if 'qwen3' in model_id.lower():
            model = Qwen3VLForConditionalGeneration.from_pretrained(f"Qwen/{model_id}", dtype="auto", device_map="auto")
            model = model.model  # unwrap from generation wrapper to get the pure backbone
            processor = AutoProcessor.from_pretrained(f"Qwen/{model_id}")
            transform = HFChatImagePreprocessor(
                processor,
                fixed_prompt=fixed_prompt,
                add_generation_prompt=add_generation_prompt,
            )
            
        # elif 'ministra' in model_id.lower():
        #     model = Mistral3ForConditionalGeneration.from_pretrained(model_id, dtype="auto", device_map="auto")
        #     processor = MistralCommonBackend.from_pretrained(model_id)
        #     transform = HFChatImagePreprocessor(processor)
        
        elif 'vjepa2' in model_id.lower():
            model = AutoModel.from_pretrained(f"facebook/{model_id}")
            model = model.encoder  # predictor is not needed
            processor = AutoVideoProcessor.from_pretrained(f"facebook/{model_id}")
            transform = HFVideoPreprocessor(processor)
        else: # Fallback to generic AutoModel
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
            processor = AutoProcessor.from_pretrained(model_id)
            transform = HFChatImagePreprocessor(
                processor,
                fixed_prompt=fixed_prompt,
                add_generation_prompt=add_generation_prompt,
            )
            
        
    except Exception as e:
        raise ValueError(f"Error loading model '{model_id}': {e}")
    
    
    return model, transform
