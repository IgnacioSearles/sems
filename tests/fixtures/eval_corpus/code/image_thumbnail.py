from pathlib import Path

from PIL import Image, ImageOps

MAX_EDGE = 320


def make_thumbnail(source: Path, destination_dir: Path) -> Path:
    with Image.open(source) as picture:
        picture = ImageOps.exif_transpose(picture)
        picture.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
        target = destination_dir / f"{source.stem}_thumb.jpg"
        picture.convert("RGB").save(target, "JPEG", quality=85)
    return target


def thumbnail_folder(folder: Path) -> list[Path]:
    output = folder / "thumbs"
    output.mkdir(exist_ok=True)
    return [make_thumbnail(path, output) for path in folder.glob("*.jpg")]
