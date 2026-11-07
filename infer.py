"""Ask a question about one or more local images in the supplied order."""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from experiment import MODEL, encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter")
    args = parser.parse_args()
    content = [{"type": "image", "image": str(path.resolve(strict=True))} for path in args.images]
    content.append({"type": "text", "text": args.question})
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                            device_map={"": "cuda"}, attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    batch = encode(processor, [{"role": "user", "content": content}], True)
    with torch.inference_mode():
        ids = model.eval().generate(**batch, do_sample=False, max_new_tokens=64)
    print(processor.decode(ids[0, batch.input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
