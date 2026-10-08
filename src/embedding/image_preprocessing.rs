//! Port of the Hugging Face `Gemma4ImageProcessor`: aspect-preserving resize, rescale to [0, 1],
//! patchify, and pad to a fixed patch budget.

use fast_image_resize::images::Image as ResizeBuffer;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, RgbImage, Rgba};

use super::EmbeddingError;
use super::config::VisionConfig;

const CHANNELS: usize = 3;
const PADDING_POSITION: i64 = -1;

/// Vision encoder input for a single image, padded to `max_patches`.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    /// `[max_patches, patch_size * patch_size * 3]`, row-major, channels interleaved per pixel.
    pub pixel_values: Vec<f32>,
    /// `[max_patches, 2]` as `(x, y)` patch coordinates; padding rows are `(-1, -1)`.
    pub position_ids: Vec<i64>,
    pub max_patches: usize,
    pub patch_pixels: usize,
    /// Number of soft tokens the vision encoder will emit for this image.
    pub soft_token_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetSize {
    pub width: usize,
    pub height: usize,
}

pub struct ImagePreprocessor {
    patch_size: usize,
    pooling_kernel_size: usize,
    max_patches: usize,
}

impl ImagePreprocessor {
    pub fn new(config: &VisionConfig) -> Self {
        Self {
            patch_size: config.patch_size,
            pooling_kernel_size: config.pooling_kernel_size,
            max_patches: config.max_soft_tokens * config.pooling_kernel_size.pow(2),
        }
    }

    pub fn preprocess(&self, image: &DynamicImage) -> Result<PreprocessedImage, EmbeddingError> {
        let rgb = flatten_onto_white(image);
        let target = self.target_size(rgb.width() as usize, rgb.height() as usize)?;
        let resized = resize_bicubic(&rgb, target)?;
        Ok(self.patchify(&resized, target))
    }

    /// Largest aspect-preserving size within the patch budget whose sides are multiples of
    /// `pooling_kernel_size * patch_size`. Mirrors `get_aspect_ratio_preserving_size` exactly,
    /// including its handling of extreme aspect ratios.
    pub fn target_size(&self, width: usize, height: usize) -> Result<TargetSize, EmbeddingError> {
        if width == 0 || height == 0 {
            return Err(EmbeddingError::InvalidImage(format!("image has zero size ({width}x{height})")));
        }
        let target_pixels = (self.max_patches * self.patch_size.pow(2)) as f64;
        let scale = (target_pixels / (width * height) as f64).sqrt();
        let side_multiple = self.pooling_kernel_size * self.patch_size;
        let round_down = |ideal: f64| (ideal / side_multiple as f64).floor() as usize * side_multiple;

        let mut target_width = round_down(scale * width as f64);
        let mut target_height = round_down(scale * height as f64);
        let max_side = (self.max_patches / self.pooling_kernel_size.pow(2)) * side_multiple;

        match (target_width, target_height) {
            (0, 0) => {
                return Err(EmbeddingError::InvalidImage(format!(
                    "{width}x{height} cannot be resized to a non-empty patch grid"
                )));
            }
            (_, 0) => {
                target_height = side_multiple;
                target_width = ((width / height) * side_multiple).min(max_side);
            }
            (0, _) => {
                target_width = side_multiple;
                target_height = ((height / width) * side_multiple).min(max_side);
            }
            _ => {}
        }

        if (target_width * target_height) as f64 > target_pixels {
            return Err(EmbeddingError::InvalidImage(format!(
                "resizing {width}x{height} to {target_width}x{target_height} exceeds the patch budget"
            )));
        }
        Ok(TargetSize { width: target_width, height: target_height })
    }

    fn patchify(&self, image: &RgbImage, size: TargetSize) -> PreprocessedImage {
        let patch_size = self.patch_size;
        let patch_pixels = patch_size * patch_size * CHANNELS;
        let patches_wide = size.width / patch_size;
        let patches_high = size.height / patch_size;
        let real_patches = patches_wide * patches_high;

        let mut pixel_values = vec![0.0_f32; self.max_patches * patch_pixels];
        let mut position_ids = vec![PADDING_POSITION; self.max_patches * 2];
        let raw = image.as_raw();

        for patch_row in 0..patches_high {
            for patch_column in 0..patches_wide {
                let patch_index = patch_row * patches_wide + patch_column;
                let patch = &mut pixel_values[patch_index * patch_pixels..(patch_index + 1) * patch_pixels];
                for row_in_patch in 0..patch_size {
                    let source_row = patch_row * patch_size + row_in_patch;
                    let source_start = (source_row * size.width + patch_column * patch_size) * CHANNELS;
                    let source = &raw[source_start..source_start + patch_size * CHANNELS];
                    let destination = &mut patch[row_in_patch * patch_size * CHANNELS..][..patch_size * CHANNELS];
                    for (value, &byte) in destination.iter_mut().zip(source) {
                        *value = f32::from(byte) / 255.0;
                    }
                }
                position_ids[patch_index * 2] = patch_column as i64;
                position_ids[patch_index * 2 + 1] = patch_row as i64;
            }
        }

        PreprocessedImage {
            pixel_values,
            position_ids,
            max_patches: self.max_patches,
            patch_pixels,
            soft_token_count: real_patches / self.pooling_kernel_size.pow(2),
        }
    }
}

/// Matches `transformers.image_transforms.convert_to_rgb`: transparent pixels composite onto white.
fn flatten_onto_white(image: &DynamicImage) -> RgbImage {
    if !image.color().has_alpha() {
        return image.to_rgb8();
    }
    let rgba = image.to_rgba8();
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let Rgba([red, green, blue, alpha]) = *rgba.get_pixel(x, y);
        let blend = |channel: u8| {
            let alpha = u16::from(alpha);
            ((u16::from(channel) * alpha + 255 * (255 - alpha) + 127) / 255) as u8
        };
        image::Rgb([blend(red), blend(green), blend(blue)])
    })
}

/// Antialiased bicubic (Catmull-Rom, a = -0.5) convolution, matching PIL / torchvision within ±1.
///
/// Resizes in 16-bit on purpose: for 8-bit pixels fast_image_resize runs the vertical pass first,
/// while PIL runs horizontal first and clips in between. At saturated high-contrast edges (text in
/// screenshots) that ordering difference reached 17/255. The 16-bit path runs horizontal first.
fn resize_bicubic(image: &RgbImage, target: TargetSize) -> Result<RgbImage, EmbeddingError> {
    if image.width() as usize == target.width && image.height() as usize == target.height {
        return Ok(image.clone());
    }
    let invalid = |error: &dyn std::fmt::Display| EmbeddingError::InvalidImage(error.to_string());
    let widened: Vec<u8> = image.as_raw().iter().flat_map(|&value| (u16::from(value) * 257).to_ne_bytes()).collect();
    let source =
        ResizeBuffer::from_vec_u8(image.width(), image.height(), widened, PixelType::U16x3).map_err(|e| invalid(&e))?;
    let mut destination = ResizeBuffer::new(target.width as u32, target.height as u32, PixelType::U16x3);
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    Resizer::new().resize(&source, &mut destination, &options).map_err(|e| invalid(&e))?;

    let narrowed: Vec<u8> = destination
        .into_vec()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&bytes| {
            let value = u32::from(u16::from_ne_bytes(bytes));
            ((value * 255 + 32_767) / 65_535) as u8
        })
        .collect();
    RgbImage::from_raw(target.width as u32, target.height as u32, narrowed)
        .ok_or_else(|| invalid(&"resized buffer has unexpected length"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preprocessor() -> ImagePreprocessor {
        ImagePreprocessor::new(&VisionConfig { patch_size: 16, pooling_kernel_size: 3, max_soft_tokens: 280 })
    }

    #[test]
    fn target_size_matches_upstream_for_landscape_image() {
        // Reference: Gemma4ImageProcessor resizes 640x480 to 912x672 (266 soft tokens).
        let size = preprocessor().target_size(640, 480).unwrap();
        assert_eq!(size, TargetSize { width: 912, height: 672 });
    }

    #[test]
    fn target_size_handles_extreme_aspect_ratio() {
        let size = preprocessor().target_size(10_000, 10).unwrap();
        assert_eq!(size.height, 48);
        assert!(size.width * size.height <= 2520 * 256);
    }

    #[test]
    fn target_size_rejects_empty_image() {
        assert!(preprocessor().target_size(0, 100).is_err());
    }

    #[test]
    fn patchify_lays_out_positions_row_major_and_pads() {
        let image = DynamicImage::new_rgb8(640, 480);
        let output = preprocessor().preprocess(&image).unwrap();
        assert_eq!(output.soft_token_count, 266);
        assert_eq!(&output.position_ids[..6], &[0, 0, 1, 0, 2, 0]);
        let first_padding = 2394 * 2;
        assert_eq!(&output.position_ids[first_padding..first_padding + 2], &[-1, -1]);
    }

    #[test]
    fn transparent_pixels_become_white() {
        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(1, 1, Rgba([0, 0, 0, 0])));
        assert_eq!(flatten_onto_white(&image).get_pixel(0, 0).0, [255, 255, 255]);
    }
}
