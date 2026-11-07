"""Run base evaluation, response-only LoRA training, then evaluate a reloaded adapter."""
import argparse
import json
import platform
import re
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

OUT = Path("results")
MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def read(split):
    return [json.loads(line) for line in Path(f"data/{split}/rows.jsonl").read_text().splitlines()]


def encode(processor, messages, prompt):
    batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=prompt,
                                          return_dict=True, return_tensors="pt")
    return batch.to("cuda")


def evaluate(model, processor, rows, name):
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    predictions = []
    for row in rows:
        messages = row["messages"]
        batch = encode(processor, messages[:1], True)
        with torch.inference_mode():
            ids = model.generate(**batch, max_new_tokens=16, do_sample=False)
        answer = processor.decode(ids[0, batch.input_ids.shape[1]:], skip_special_tokens=True).strip()
        target = messages[1]["content"][0]["text"]
        choices = "YES|NO" if row["task"] == "pair" else "LEFT|RIGHT|ABOVE|BELOW"
        matches = set(re.findall(r"\b(" + choices + r")\b", answer.upper()))
        predictions.append(dict(id=row["id"], task=row["task"], gold=target, prediction=answer,
                                semantic=matches == {target}, exact=answer == target,
                                format=bool(re.fullmatch(choices, answer)),
                                image_tokens=int((batch.input_ids == model.config.image_token_id).sum())))
    (OUT / f"{name}.json").write_text(json.dumps(predictions, indent=2))
    return peak()


def peak():
    return round(torch.cuda.max_memory_allocated() / 2**30, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    torch.manual_seed(42)
    OUT.mkdir(exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            device_map={"": "cuda"}, attn_implementation="sdpa")
    test = read("test")
    base_peak = evaluate(model, processor, test, "base")
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for i, row in enumerate(read("train"), 1):
        messages = row["messages"]
        prefix = encode(processor, messages[:1], True).input_ids
        batch = encode(processor, messages, False)
        n = prefix.shape[1]
        assert torch.equal(prefix, batch.input_ids[:, :n])
        labels = batch.input_ids.clone()
        labels[:, :n] = -100
        loss = model(**batch, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        (loss / 2).backward()
        losses.append(round(loss.item(), 5))
        if i % 2 == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if i % 8 == 0:
            print("train", i, losses[-1], flush=True)
    train_peak = peak()
    model.save_pretrained("adapter", save_embedding_layers=False)
    # Local filesystem paths are machine-specific.
    path = Path("adapter/adapter_config.json")
    cfg = json.loads(path.read_text())
    cfg["base_model_name_or_path"] = MODEL
    path.write_text(json.dumps(cfg, indent=2))
    Path("adapter/README.md").write_text('# Spatial instruction LoRA\n\nBase model: Qwen/Qwen3-VL-2B-Instruct. Training settings, evaluation and usage are in the [project README](../README.md).\n')
    backbone = model.unload()
    del optimizer, params, loss, model
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(backbone, "adapter")
    tuned_peak = evaluate(model, processor, test, "lora")
    metrics = dict(model=MODEL, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   transformers=transformers.__version__, peft=peft.__version__, python=platform.python_version(),
                   base_peak_GiB=base_peak, train_peak_GiB=train_peak, lora_peak_GiB=tuned_peak, loss=losses)
    (OUT / "run.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
