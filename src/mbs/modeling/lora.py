from typing import List
import re

import torch
from peft import LoraConfig, get_peft_model, PeftModel



def build_lora_regex(target_modules: list, prefix: str = "backbone") -> str:
    """
    Builds a regex string for PEFT to target specific modules within a given prefix.
    """
    if not target_modules:
        raise ValueError("You must provide at least one target module.")
    
    # Escape the module names just in case they contain regex special characters
    escaped_modules = [re.escape(mod) for mod in target_modules]
    
    # Join them together with the OR operator: (qkv|fc1|fc2)
    modules_group = "|".join(escaped_modules)
    
    # Construct the final raw string pattern
    regex_pattern = rf".*{re.escape(prefix)}.*\.\b({modules_group})\b"
    
    return regex_pattern

def create_lora_model(
    model: torch.nn.Module,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_bias: str = "none",
    lora_target_modules: List[str] = [],
    lora_modules_to_save: List[str] = [],
    lora_exclude_modules: List[str] = []
) -> PeftModel:
    
    """
    Create a LoRA model based on the given base model and LoRA configuration.

    Args:
        model (torch.nn.Module): The base model to which LoRA will be applied.
        lora_r (int, optional): The rank of the LoRA decomposition. Default is 8.
        lora_alpha (int, optional): The scaling factor for the LoRA updates. Default is 32.
        lora_dropout (float, optional): The dropout rate for the LoRA layers. Default is 0.1.
        lora_target_modules (List[str], optional): A list of module names to which LoRA will be applied. Default is an empty list.
        lora_modules_to_save (List[str], optional): A list of module names whose parameters will be saved during training. Default is an empty list.
        lora_exclude_modules (List[str], optional): A list of module names to exclude from LoRA. Default is an empty list.

    Returns:
        torch.nn.Module: The modified model with LoRA applied.
    """
    peft_config = LoraConfig(
        r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=build_lora_regex(lora_target_modules),
        modules_to_save=lora_modules_to_save,
        exclude_modules=lora_exclude_modules
    )
    print("Creating LoRA model with config:", peft_config)
    lora_model = get_peft_model(model, peft_config)
    return lora_model