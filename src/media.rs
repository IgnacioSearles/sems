//! Recognizing and decoding images for indexing.

use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, ImageDecoder, ImageReader};

/// Formats the `image` crate decodes. HEIC (iPhone default) is not supported yet.
const IMAGE_EXTENSIONS: [&str; 9] = ["jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff", "jfif"];

/// Images whose shorter side is below this are icons, sprites or thumbnails: not worth the
/// seconds each image costs to embed on a CPU, and rarely what anyone searches for.
pub const MIN_IMAGE_EDGE: u32 = 64;

pub fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| IMAGE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
}

#[derive(Debug)]
pub enum LoadedImage {
    Image(DynamicImage),
    TooSmall,
}

/// Decodes an image upright: phone cameras store pixels sideways plus an EXIF orientation tag,
/// and a sideways photo embeds noticeably worse.
pub fn load_image(path: &Path) -> Result<LoadedImage> {
    let reader = ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut decoder = reader.into_decoder().with_context(|| format!("unsupported image {}", path.display()))?;
    let (width, height) = decoder.dimensions();
    if width.min(height) < MIN_IMAGE_EDGE {
        return Ok(LoadedImage::TooSmall);
    }
    let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut image =
        DynamicImage::from_decoder(decoder).with_context(|| format!("failed to decode {}", path.display()))?;
    image.apply_orientation(orientation);
    Ok(LoadedImage::Image(image))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_image_extensions_case_insensitively() {
        assert!(is_image_path(Path::new("holiday/IMG_0042.JPG")));
        assert!(is_image_path(Path::new("scan.tiff")));
        assert!(!is_image_path(Path::new("notes.md")));
        assert!(!is_image_path(Path::new("Makefile")));
    }

    #[test]
    fn small_images_are_reported_without_decoding() {
        let directory = tempfile::tempdir().unwrap();
        let icon = directory.path().join("icon.png");
        image::RgbImage::new(32, 200).save(&icon).unwrap();
        assert!(matches!(load_image(&icon).unwrap(), LoadedImage::TooSmall));
    }

    #[test]
    fn loads_regular_images() {
        let directory = tempfile::tempdir().unwrap();
        let photo = directory.path().join("photo.png");
        image::RgbImage::new(120, 80).save(&photo).unwrap();
        let LoadedImage::Image(image) = load_image(&photo).unwrap() else { panic!("expected an image") };
        assert_eq!((image.width(), image.height()), (120, 80));
    }

    #[test]
    fn corrupt_images_are_errors() {
        let directory = tempfile::tempdir().unwrap();
        let broken = directory.path().join("broken.jpg");
        std::fs::write(&broken, b"\xFF\xD8\xFF\xE0 not really a jpeg").unwrap();
        assert!(load_image(&broken).is_err());
    }
}
