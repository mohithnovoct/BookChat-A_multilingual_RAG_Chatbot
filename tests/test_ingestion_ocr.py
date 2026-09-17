import unittest

from llama_index.core import Document
from PIL import Image

from bookchat.core.ingestion import (
    _clean_multilingual_text,
    _is_valid_multilingual_content,
    _preprocess_ocr_image,
    detect_doc_lang,
    detect_tesseract_lang,
    get_chunks,
)


class IngestionOcrTests(unittest.TestCase):
    def test_clean_normalizes_danda_spacing_and_hyphen_linebreaks(self):
        raw = "hello-\nworld ।  next"
        cleaned = _clean_multilingual_text(raw)
        self.assertIn("helloworld", cleaned)
        self.assertIn("। ", cleaned)

    def test_valid_content_accepts_indic_and_rejects_short_garbage(self):
        kannada = "ಕನ್ನಡ ಭಾಷೆಯ ಪಠ್ಯ ಇದು ಪರೀಕ್ಷೆಗಾಗಿ ಬರೆಯಲಾಗಿದೆ ಮತ್ತು ಸಾಕಷ್ಟು ಉದ್ದವಾಗಿದೆ."
        self.assertTrue(_is_valid_multilingual_content(kannada))
        self.assertFalse(_is_valid_multilingual_content("???"))
        self.assertFalse(_is_valid_multilingual_content("short"))

    def test_language_detection_for_tesseract_and_metadata(self):
        kannada = "ಕನ್ನಡ " * 10
        punjabi = "ਪੰਜਾਬੀ " * 10
        english = "This is a long enough English sample for script detection."
        self.assertEqual(detect_doc_lang(kannada), "kn")
        self.assertEqual(detect_doc_lang(punjabi), "pa")
        self.assertEqual(detect_doc_lang(english), "en")
        self.assertEqual(detect_tesseract_lang(kannada), "kan")
        self.assertEqual(detect_tesseract_lang(punjabi), "pan")
        self.assertEqual(detect_tesseract_lang(english), "eng")
        self.assertIn("+", detect_tesseract_lang(""))

    def test_preprocess_converts_to_binary_grayscale(self):
        image = Image.new("RGB", (32, 32), color=(120, 80, 80))
        processed = _preprocess_ocr_image(image)
        self.assertEqual(processed.mode, "L")
        pixels = list(processed.get_flattened_data()) if hasattr(processed, "get_flattened_data") else list(processed.getdata())
        self.assertTrue(all(pixel in (0, 255) for pixel in pixels))

    def test_chunker_assigns_lang_and_chunk_ids(self):
        text = (
            "First sentence about the river. Second sentence continues the story. "
            "Third sentence adds more English context for splitting."
        )
        nodes = get_chunks(
            [Document(text=text, metadata={"filename": "book.txt"})],
            chunk_size=80,
            chunk_overlap=20,
        )
        self.assertGreaterEqual(len(nodes), 1)
        self.assertEqual(nodes[0].metadata["filename"], "book.txt")
        self.assertEqual(nodes[0].metadata["lang"], "en")
        self.assertEqual(nodes[0].metadata["chunk_id"], 0)
        if len(nodes) > 1:
            self.assertEqual(nodes[1].metadata["chunk_id"], 1)


if __name__ == "__main__":
    unittest.main()
