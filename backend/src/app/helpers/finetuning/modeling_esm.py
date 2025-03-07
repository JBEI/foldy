# my_esm_lib/modeling_esm.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, inject_adapter_in_model


def load_esm_model(
    checkpoint: str,
    num_labels: int = 1,
    half_precision: bool = False,
    train_full: bool = False,
    deepspeed: bool = False,
):
    """
    Loads an ESM checkpoint + classification head with `num_labels`.
    Optionally uses LoRA. If `train_full=True`, we do not freeze the base
    model. Otherwise, we freeze the base model and only train LoRA.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if half_precision and deepspeed:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels, torch_dtype=torch.float16
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels
        )

    if train_full:
        # Train all parameters
        for param in model.parameters():
            param.requires_grad = True
        return model, tokenizer

    # Otherwise, inject LoRA (PEFT)
    peft_config = LoraConfig(
        r=4,
        lora_alpha=1,
        bias="all",
        target_modules=["query", "key", "value", "dense"],  # adapt as needed
    )
    model = inject_adapter_in_model(peft_config, model)

    # Now unfreeze the classification head + LoRA only
    # (PEFT code by default sets the LoRA layers requires_grad=True)
    # But if you need to ensure all other layers are frozen:
    for name, param in model.named_parameters():
        # If parameter is not in LoRA or classifier, freeze it
        if "lora_" not in name and "classifier" not in name:
            param.requires_grad = False

    return model, tokenizer
