import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from bookchat.api.schemas import QueryRequest, QueryResponse, SourceReference
from bookchat.config import FALLBACK_ANSWER, LLM_MAX_NEW_TOKENS
from bookchat.core.generate import (
    HuggingFaceInferenceLLM,
    build_system_prompt,
    ndcg_at_k,
    parse_model_answer,
    recall_at_k,
)


class LanguageContractTests(unittest.TestCase):
    def test_defaults_are_english_answer_and_automatic_query_language(self):
        request = QueryRequest(question="What is the main theme?")

        self.assertEqual(request.answer_language, "en")
        self.assertEqual(request.query_language, "auto")
        self.assertEqual(request.k, 8)

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
        self.assertIn("<answer>", prompt)

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

        llm = HuggingFaceInferenceLLM(
            model_name="test-model",
            token="test-token",
            system_prompt=build_system_prompt(),
        )
        llm._client = FakeClient()
        response = llm.complete("question")

        self.assertEqual(response.text, "answer")
        self.assertEqual(calls[0]["max_tokens"], LLM_MAX_NEW_TOKENS)
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertNotIn("repetition_penalty", calls[0])
        self.assertEqual(calls[0]["messages"][0]["role"], "system")
        self.assertIn("respond in English only", calls[0]["messages"][0]["content"])
        self.assertEqual(calls[0]["messages"][1]["role"], "user")

    def test_parse_model_answer_extracts_xml_and_rejects_indic_heavy_output(self):
        self.assertEqual(
            parse_model_answer("<answer>The river is sacred.</answer>"),
            "The river is sacred.",
        )
        self.assertEqual(parse_model_answer(""), FALLBACK_ANSWER)
        kannada = "ಕನ್ನಡ " * 20
        self.assertEqual(parse_model_answer(f"<answer>{kannada}</answer>"), FALLBACK_ANSWER)


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_and_ndcg_on_golden_orderings(self):
        retrieved = ["en-theme", "other", "kn-theme"]
        self.assertEqual(recall_at_k(retrieved, ["en-theme"], 20), 1.0)
        self.assertGreater(ndcg_at_k(retrieved, ["en-theme"], 20), 0.9)
        self.assertEqual(recall_at_k(["miss"], ["en-theme"], 20), 0.0)


if __name__ == "__main__":
    unittest.main()
