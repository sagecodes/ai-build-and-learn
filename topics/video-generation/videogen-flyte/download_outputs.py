"""Download a run's artifact outputs (the generated .mp4 clips) out of the Flyte
devbox blob store to your local disk.

Walks EVERY action in the run (not just the root), so it works for `compare` and
`animate` runs (the clips live in the child `generate_for_model` actions) as well
as `generate_one`. It skips the weight-fetch tasks and only pulls media files, so
the 30-126GB `.safetensors` Dirs never come down, which matters a lot more here
than in the image demo, where a stray full pull was merely annoying.

The clips in the report are downscaled previews; these are the full-resolution
originals, with LTX-2's audio track intact.

The devbox blob store is rustfs (S3-compatible) with static creds. The client has
no storage creds by default, so this sets the FLYTE_AWS_* env the SDK storage layer
reads, then pulls each artifact via the Flyte download API.

Usage:
    python download_outputs.py <run_name>
    python download_outputs.py <run_name> --dest ./out
    python download_outputs.py <run_name> --all            # every file, not just media
    python download_outputs.py <run_name> --ext .mp4       # clips only
    # from a laptop (not on the Spark), hit the Tailscale IP to skip the 30002 forward:
    python download_outputs.py <run_name> --endpoint http://100.121.165.36:30002
"""
import asyncio
import base64
import json
import os
from pathlib import Path

import click

_HERE = Path(__file__).parent
_CONFIG = _HERE / ".flyte" / "config.yaml"
_SCHEMES = ("s3://", "gs://", "abfs://", "abfss://")
# Media artifacts we care about; weights/configs (.safetensors/.json/...) are skipped.
_MEDIA_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v",
    ".wav", ".mp3", ".flac", ".ogg", ".opus",
}


def _collect_uris(node, out: set):
    """Recursively collect blob URIs, decoding any msgpack dataclass literals."""
    if isinstance(node, str):
        if node.startswith(_SCHEMES):
            out.add(node)
    elif isinstance(node, dict):
        binary = node.get("binary")
        if isinstance(binary, dict) and binary.get("tag") == "msgpack" and binary.get("value"):
            try:
                import msgpack

                _collect_uris(msgpack.unpackb(base64.b64decode(binary["value"]), raw=False), out)
            except Exception:
                pass
        for v in node.values():
            _collect_uris(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _collect_uris(v, out)


async def _pull(uri: str, dest: Path, exts: set, allow_all: bool) -> list[str]:
    """Download a Dir (recursively) or single File URI, keeping only wanted files."""
    import flyte.io

    base = uri.rstrip("/")
    subdir = dest / base.rsplit("/", 1)[-1]
    saved: list[str] = []

    def _wanted(name: str) -> bool:
        return allow_all or Path(name).suffix.lower() in exts

    # Try as a directory first (recursive walk).
    try:
        walked = False
        async for f in flyte.io.Dir(path=uri).walk():
            walked = True
            rel = f.path[len(base):].lstrip("/") if f.path.startswith(base) else f.path.rsplit("/", 1)[-1]
            if not _wanted(rel):
                continue
            target = subdir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            await f.download(str(target))
            saved.append(str(target))
        if walked:
            return saved
    except Exception:
        pass

    # Fall back to a single file.
    if _wanted(base):
        target = dest / base.rsplit("/", 1)[-1]
        try:
            await flyte.io.File(path=uri).download(str(target))
            saved.append(str(target))
        except Exception:
            pass
    return saved


@click.command()
@click.argument("run_name")
@click.option("--dest", default="./downloads", help="Local directory to download into.")
@click.option("--all", "allow_all", is_flag=True, help="Download every file, not just media.")
@click.option("--ext", default="", help="Comma-separated extensions to keep (e.g. .mp4,.png).")
@click.option("--skip-uri", default="/repo",
              help="Comma-separated URI substrings to skip (weight-fetch '/repo' dirs).")
@click.option("--endpoint", default="http://localhost:30002",
              help="rustfs S3 endpoint. Use http://<tailscale-ip>:30002 from a laptop.")
@click.option("--access-key", default="rustfs")
@click.option("--secret-key", default="rustfsstorage")
def main(run_name, dest, allow_all, ext, skip_uri, endpoint, access_key, secret_key):
    """Download artifact outputs of RUN_NAME (all tasks) from the devbox to --dest."""
    # Storage auth for the SDK (client has none by default). Respect anything the
    # user already exported; otherwise fall back to the devbox rustfs defaults.
    os.environ.setdefault("FLYTE_AWS_ENDPOINT", endpoint)
    os.environ.setdefault("FLYTE_AWS_ACCESS_KEY_ID", access_key)
    os.environ.setdefault("FLYTE_AWS_SECRET_ACCESS_KEY", secret_key)
    os.environ.setdefault("FLYTE_AWS_S3_ADDRESSING_STYLE", "path")

    import flyte
    from flyte.remote import ActionDetails, RunDetails
    from flyte.remote._action import Action

    flyte.init_from_config(str(_CONFIG))
    dest_path = Path(dest).absolute()
    dest_path.mkdir(parents=True, exist_ok=True)

    exts = {e if e.startswith(".") else f".{e}" for e in (x.strip().lower() for x in ext.split(",")) if e} or _MEDIA_EXTS
    skips = [s.strip() for s in skip_uri.split(",") if s.strip()]

    async def run():
        uris: set = set()

        # Root run output (covers generate_one, whose ModelRun is the root output).
        try:
            rd = await RunDetails.get.aio(name=run_name)
            _collect_uris(json.loads((await rd.outputs()).to_json()), uris)
        except Exception:
            pass

        # Every action in the run (covers compare's per-model child actions).
        async for a in Action.listall.aio(for_run_name=run_name):
            try:
                ad = await ActionDetails.get.aio(run_name=run_name, name=a.name)
                _collect_uris(json.loads((await ad.outputs()).to_json()), uris)
            except Exception:
                continue  # action not finished / has no outputs

        # Drop intermediate/weight dirs (fetch_weights outputs a '/repo' Dir).
        uris = {u for u in uris if not any(s in u for s in skips)}

        if not uris:
            click.echo("No artifact (File/Dir) outputs found for this run.")
            return

        click.echo(f"Found {len(uris)} artifact URI(s):")
        for u in sorted(uris):
            click.echo(f"  {u}")

        files: list[str] = []
        for u in sorted(uris):
            files.extend(await _pull(u, dest_path, exts, allow_all))

        files = sorted(set(files))
        click.echo(f"\nDownloaded {len(files)} file(s) to {dest_path}:")
        for f in files:
            click.echo(f"  {os.path.getsize(f):>10,} B  {f}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
