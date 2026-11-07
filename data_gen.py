"""Generate ordered image conversations and deterministic, balanced splits."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

DIRECTIONS = {"left": (-1, 0), "right": (1, 0), "above": (0, -1), "below": (0, 1)}


def scene(path, direction, rng):
    image = Image.new("RGB", (224, 224), (247, 246, 240))
    draw = ImageDraw.Draw(image)
    dx, dy = DIRECTIONS[direction]
    cx, cy = (rng.randint(93, 131) for _ in range(2))
    distance = rng.randint(36, 52)
    for shape, sign in [("circle", 1), ("square", -1)]:
        x = cx + sign * distance * dx + rng.randint(-6, 6)
        y = cy + sign * distance * dy + rng.randint(-6, 6)
        size = rng.randint(15, 25)
        box = (x-size, y-size, x+size, y+size)
        if shape == "circle":
            draw.ellipse(box, fill=(202, 43, 53))
        else:
            draw.rectangle(box, fill=(38, 89, 189))
    image.save(path)


def generate(split, count, seed):
    folder = Path("data") / split
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        direction = list(DIRECTIONS)[(i // 2) % 4]
        pair = i % 2 == 1
        changed = (i // 2 + i // 8) % 2 == 0
        directions = [direction]
        if pair:
            alternatives = [d for d in DIRECTIONS if d != direction]
            directions.append(rng.choice(alternatives) if changed else direction)
        content = []
        for j, d in enumerate(directions):
            path = folder / f"{i:03d}-{j}.png"
            scene(path, d, rng)
            content.append({"type": "image", "image": path.as_posix()})
        if pair:
            question = "Between image 1 and image 2, did the red circle change sides relative to the blue square? Answer exactly YES or NO."
            answer = "YES" if changed else "NO"
        else:
            question = "Where is the red circle relative to the blue square? Answer exactly LEFT, RIGHT, ABOVE or BELOW."
            answer = direction.upper()
        content.append({"type": "text", "text": question})
        rows.append({"id": f"{split}-{i:03d}", "task": "pair" if pair else "single",
                     "messages": [{"role": "user", "content": content},
                                  {"role": "assistant", "content": [{"type": "text", "text": answer}]}]})
    rng.shuffle(rows)
    (folder / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")


if __name__ == "__main__":
    generate("train", 64, 1701)
    generate("test", 32, 3177)
