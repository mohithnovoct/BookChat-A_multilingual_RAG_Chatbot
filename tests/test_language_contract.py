import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from bookchat.api.schemas import QueryRequest, QueryResponse, SourceReference
from bookchat.core.generate import HuggingFaceInferenceLLM, build_system_prompt


class LanguageContractTests(unittest.TestCase):
    def test_defaults_are_english_answer_and_automatic_query_language(self):
        request = QueryRequest(question="What is the main theme?")

        self.assertEqual(request.answer_language, "en")
        self.assertEqual(request.query_language, "auto")

    def test_retired_lang_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            QueryRequest(question="What happened?", lang="kn")

    def test_non_english_answer_language_is_rejected(self):
        with self.assertRaises(ValidationError):
            QueryRequest(question="What happened?", answer_language="kn")

    def test_prompt_allows_multilingual_context_but_requires_english(self):
        prompt = build_system_prompt()

        self.assertIn("respond in English only", prompt)
        self.assertIn("English, Kannada, Punjabi", prompt)
        self.assertIn("Information not found in the source documents.", prompt)

    def test_sources_are_serializable(self):
        response = QueryResponse(
            question="What happened?",
            answer="The event occurred.",
            sources=[SourceReference(filename="book.pdf", page=3, chunk_id=2)],
        )

        self.assertEqual(response.model_dump()["sources"][0]["page"], 3)

    def test_huggingface_chat_request_uses_supported_arguments(self):
        calls = []

        class FakeClient:
            def chat_completion(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
                )

        llm = HuggingFaceInferenceLLM(model_name="test-model", token="test-token")
        llm._client = FakeClient()
        response = llm.complete("question")

        self.assertEqual(response.text, "answer")
        self.assertEqual(calls[0]["max_tokens"], 1500)
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertNotIn("repetition_penalty", calls[0])


if __name__ == "__main__":
    unittest.main()