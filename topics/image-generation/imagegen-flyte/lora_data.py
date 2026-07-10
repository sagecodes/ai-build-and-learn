"""Small captioned image datasets for LoRA fine-tuning, and their preprocessing.

Flyte-free, and `torch`/`datasets` are imported inside the functions, so an
orchestrator pod can import this module just to read the registry.

Two flavours of LoRA live in this repo:

  subject : every image is the same thing, tied to one fixed prompt containing a
            rare token ("a photo of sks dog"). Teaches an *object*. That's what
            lora_finetune.py does to SDXL.
  style   : varied subjects sharing one look, per-image captions plus a trigger
            phrase ("yarn art style"). Teaches an *aesthetic*. That's what
            lora_chroma.py does to Chroma.

The size of the before/after depends on how rare the trigger is, NOT on how
striking the style is. Measured on Chroma: `yarn art style` is plain English the
base model already renders well, so the adapter mostly transfers the training
photos' dark studio backdrop (near-black pixels went 2% -> 60%) rather than
teaching yarn. A rare token carries no prior, so the base model simply cannot
draw it and the contrast is unmistakable. That is what DreamBooth's `sks` is for,
and why `trtcrd` and `3dicon` below are the better demos.

Every repo below was checked to exist on the Hub. Counts are rows, not bytes;
these are all tiny. `license` is what the dataset page declares, and
"unspecified" means the page carries no license tag at all: fine for a demo,
worth checking before anything commercial.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Caption columns, in the order we'd rather have them. Different uploaders pick
# different names, and a couple of sets ship two captions of differing quality
# (tuxemon has a weak BLIP one and a good GPT-4 one), so order matters.
CAPTION_COLUMNS = ("gpt4_turbo_caption", "caption", "prompt", "text", "blip_caption")
IMAGE_COLUMNS = ("image", "img", "jpg", "png")


@dataclass(frozen=True)
class LoRADataset:
    key: str                        # short handle for the CLI
    repo: str                       # HuggingFace dataset repo id
    kind: str                       # "style" | "subject"
    trigger: str                    # phrase that activates the LoRA at sample time
    caption_column: str | None      # None => no captions, use `instance_prompt`
    instance_prompt: str            # the fixed prompt, for caption-less subject sets
    eval_prompts: tuple[str, ...]   # prompts that show base-vs-tuned clearly
    n_images: int
    license: str
    notes: str = ""


DATASETS: dict[str, LoRADataset] = {
    # The default. 18 images is the whole set, so a few hundred steps is a full
    # pass many times over and training finishes inside a coffee break. Captions
    # already end in "yarn art style", so the trigger is baked into the data.
    "yarn-art": LoRADataset(
        key="yarn-art",
        repo="Norod78/Yarn-art-style",
        kind="style",
        trigger="yarn art style",
        caption_column="text",
        instance_prompt="yarn art style",
        eval_prompts=(
            "a fox sitting in the snow, yarn art style",
            "an astronaut riding a horse, yarn art style",
            "the Golden Gate Bridge at sunset, yarn art style",
        ),
        n_images=18,
        license="unspecified",
        notes=("Crocheted/knitted look; the canonical HF LoRA tutorial set. VERIFIED on Chroma: "
               "the base model already does yarn art, so the adapter mostly adds the training "
               "photos' dark backdrop. Real, but subtle. Use lora_scale ~0.6, or pick a "
               "rare-token set for a dramatic before/after."),
    ),
    # Public domain, and the effect is compositional rather than just textural:
    # the tuned model draws the whole framed card, border and banner included.
    "tarot": LoRADataset(
        key="tarot",
        repo="multimodalart/1920-raider-waite-tarot-public-domain",
        kind="style",
        # The captions read "a trtcrd of a young man standing on the edge of a
        # cliff...". `trtcrd` is a deliberately rare token, exactly like DreamBooth's
        # "sks": it carries almost no prior, so the LoRA gets to define it. Spelling
        # it "tarot card" here would train on a phrase the base model already knows
        # and dilute the effect.
        trigger="trtcrd",
        caption_column="caption",
        instance_prompt="a trtcrd",
        eval_prompts=(
            "a trtcrd of an astronaut floating in space",
            "a trtcrd of a laptop computer",
            "a trtcrd of a red panda barista",
        ),
        n_images=78,
        license="public domain (1920 Rider-Waite deck)",
        notes="Cleanest license here. Teaches layout + flat 1920 palette, not just texture.",
    ),
    # Glossy 3D app-icon renders. Trigger is a single rare-ish token, so it's the
    # closest style set to how a DreamBooth subject LoRA behaves.
    "3d-icon": LoRADataset(
        key="3d-icon",
        repo="linoyts/3d_icon",
        kind="style",
        trigger="3dicon",
        caption_column="prompt",
        instance_prompt="a 3dicon",
        eval_prompts=(
            "a 3dicon of a fox, on a black background",
            "a 3dicon of a coffee cup, on a black background",
            "a 3dicon of a rocket ship, on a black background",
        ),
        n_images=23,
        license="Unsplash (free use)",
        notes="Studio-lit rounded 3D icons. Prompts must keep the 'on a black background' tail to match the data.",
    ),
    # The classic. Kept so the Chroma trainer can reproduce the same subject-LoRA
    # story lora_finetune.py tells with SDXL, on a flow-matching backbone.
    "dog": LoRADataset(
        key="dog",
        repo="diffusers/dog-example",
        kind="subject",
        trigger="sks dog",
        caption_column=None,
        instance_prompt="a photo of sks dog",
        eval_prompts=(
            "a photo of sks dog wearing a red superhero cape, cinematic",
            "a photo of sks dog sitting on the moon",
        ),
        n_images=5,
        license="unspecified",
        notes="DreamBooth's 5-photo dog. No captions; every image trains on `instance_prompt`.",
    ),
}

DEFAULT_DATASET = "yarn-art"


def get_dataset(key: str) -> LoRADataset:
    try:
        return DATASETS[key]
    except KeyError:
        raise ValueError(f"Unknown dataset {key!r}. Known: {', '.join(DATASETS)}") from None


def _pick_column(available: list[str], preferred: tuple[str, ...]) -> str | None:
    return next((c for c in preferred if c in available), None)


def caption_for(ds: LoRADataset, raw: str | None) -> str:
    """The training caption for one row: the dataset's own text plus the trigger.

    Most of these sets already bake the trigger into every caption, so we only
    append when it's genuinely missing. Doubling it ("yarn art style, yarn art
    style") teaches the model that the phrase is noise, which is the opposite of
    what a trigger is for.
    """
    if not raw:
        return ds.instance_prompt
    raw = raw.strip()
    if ds.trigger and ds.trigger.lower() not in raw.lower():
        return f"{raw}, {ds.trigger}"
    return raw


def load_examples(ds: LoRADataset, *, limit: int | None = None, split: str = "train"):
    """Pull `ds` off the Hub and return [(PIL.Image RGB, caption), ...]."""
    from datasets import load_dataset

    data = load_dataset(ds.repo, split=split)
    img_col = _pick_column(data.column_names, IMAGE_COLUMNS)
    if img_col is None:
        raise RuntimeError(
            f"{ds.repo} has no image column; saw {data.column_names}"
        )

    # Trust the registry, but fall back to sniffing if the uploader renamed the
    # column out from under us.
    cap_col = ds.caption_column if ds.caption_column in data.column_names else None
    if ds.caption_column and cap_col is None:
        cap_col = _pick_column(data.column_names, CAPTION_COLUMNS)

    if limit is not None and limit < len(data):
        data = data.select(range(limit))

    return [
        (row[img_col].convert("RGB"), caption_for(ds, row[cap_col] if cap_col else None))
        for row in data
    ]


def preprocess(img, resolution: int):
    """One PIL image to a (3, R, R) tensor in [-1, 1], center-cropped."""
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return tf(img)


# ── Materializing a dataset to a plain folder ──────────────────────────────────
#
# `load_examples` hits HuggingFace, which is CPU-and-network work. Doing it inside
# the GPU training task means an idle accelerator while a parquet file downloads,
# so a separate cached task writes the images here instead. The on-disk layout is
# deliberately boring (PNGs + one JSONL), so reading it back needs no `datasets`
# dependency and you can eyeball the folder.

CAPTIONS_FILE = "captions.jsonl"


def save_examples(examples, dest: str | Path) -> int:
    """Write [(PIL image, caption)] to `dest` as PNGs plus `captions.jsonl`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / CAPTIONS_FILE).open("w") as f:
        for i, (img, caption) in enumerate(examples):
            name = f"{i:04d}.png"
            img.save(dest / name)
            f.write(json.dumps({"file_name": name, "caption": caption}) + "\n")
    return len(examples)


def load_prepared(src: str | Path):
    """Read back what `save_examples` wrote: [(PIL image RGB, caption)]."""
    from PIL import Image

    src = Path(src)
    manifest = src / CAPTIONS_FILE
    if not manifest.exists():
        raise RuntimeError(f"{src} has no {CAPTIONS_FILE}; was it written by save_examples?")

    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"{manifest} is empty")
    return [(Image.open(src / r["file_name"]).convert("RGB"), r["caption"]) for r in rows]
