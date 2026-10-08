//! Extracting text from documents (PDF), page by page.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Once;

use anyhow::{Result, anyhow};

/// Text of each page of a PDF, in order (page 1 first). Pages without a text layer are empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdfText {
    pub pages: Vec<String>,
}

impl PdfText {
    /// Scanned documents carry their text as pixels only; they need OCR, which sems lacks.
    pub fn has_text(&self) -> bool {
        self.pages.iter().any(|page| !page.trim().is_empty())
    }
}

/// Extracts per-page text. Malformed or encrypted files are errors, never panics: pdf-extract can
/// panic on unusual fonts and broken structure, so extraction runs under `catch_unwind` with its
/// panic message suppressed, and the file is reported as unreadable.
pub fn extract_pdf_text(bytes: &[u8]) -> Result<PdfText> {
    install_quiet_panic_hook();
    EXTRACTING_PDF.set(true);
    let outcome = catch_unwind(AssertUnwindSafe(|| pdf_extract::extract_text_from_mem_by_pages(bytes)));
    EXTRACTING_PDF.set(false);
    match outcome {
        Ok(Ok(pages)) => Ok(PdfText { pages }),
        Ok(Err(error)) => Err(anyhow!("cannot read PDF: {error}")),
        Err(_) => Err(anyhow!("cannot read PDF: the parser failed on this file")),
    }
}

thread_local! {
    /// Set while this thread is inside pdf-extract, so its panics are not printed.
    static EXTRACTING_PDF: Cell<bool> = const { Cell::new(false) };
}

/// Wraps the process panic hook once: panics inside PDF extraction stay silent (they are reported
/// as unreadable files instead), every other panic still reaches the original hook.
fn install_quiet_panic_hook() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !EXTRACTING_PDF.get() {
                previous(info);
            }
        }));
    });
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval_corpus/docs").join(name)
    }

    #[test]
    fn extracts_text_page_by_page() {
        let text = extract_pdf_text(&std::fs::read(fixture("lease_agreement.pdf")).unwrap()).unwrap();
        assert_eq!(text.pages.len(), 3);
        assert!(text.pages[0].contains("monthly rent is 1,450 EUR"), "page 1: {:?}", text.pages[0]);
        assert!(text.pages[2].contains("two months' written notice"), "page 3: {:?}", text.pages[2]);
    }

    #[test]
    fn scanned_documents_have_no_text() {
        let text = extract_pdf_text(&std::fs::read(fixture("scanned_receipt.pdf")).unwrap()).unwrap();
        assert!(!text.has_text());
    }

    #[test]
    fn broken_files_are_errors_not_panics() {
        assert!(extract_pdf_text(b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 9 0 R >> endobj\n%%EOF").is_err());
        assert!(extract_pdf_text(b"not a pdf at all").is_err());
    }
}
