import unittest
from types import SimpleNamespace

from bookchat.core.generate import (
    GOLDEN_RETRIEVAL_CASES,
    QA_USER_TEMPLATE,
    build_system_prompt,
    get_rag_chain,
    ndcg_at_k,
    recall_at_k,
)


class HybridRetrievalContractTests(unittest.TestCase):
    def test_user_template_does_not_repeat_system_policy(self):
        self.assertNotIn("You MUST respond in English only", QA_USER_TEMPLATE)
        self.assertIn("{context_str}", QA_USER_TEMPLATE)
        self.assertIn("{query_str}", QA_USER_TEMPLATE)

    def test_get_rag_chain_forwards_custom_system_prompt(self):
        captured = {}

        def fake_get_query_engine(**kwargs):
            captured.update(kwargs)

            class DummyEngine:
                pass

            return DummyEngine()

        import bookchat.core.generate as generate

        original = generate.get_query_engine
        generate.get_query_engine = fake_get_query_engine
        try:
            generate.get_rag_chain(
                store=SimpleNamespace(),
                system_prompt="custom-system-prompt",
                k=8,
                query_language="kn",
            )
        finally:
            generate.get_query_engine = original

        self.assertEqual(captured["system_prompt"], "custom-system-prompt")
        self.assertEqual(captured["query_language"], "kn")
        self.assertEqual(captured["k"], 8)
        self.assertNotEqual(captured["system_prompt"], build_system_prompt())

    def test_golden_set_covers_english_kannada_punjabi(self):
        langs = {case["id"] for case in GOLDEN_RETRIEVAL_CASES}
        self.assertEqual(langs, {"en", "kn", "pa"})
        ranking = ["en-theme", "kn-theme", "pa-theme"]
        for case in GOLDEN_RETRIEVAL_CASES:
            self.assertEqual(recall_at_k(ranking, case["relevant_chunk_ids"], 20), 1.0)
            self.assertGreater(ndcg_at_k(ranking, case["relevant_chunk_ids"], 20), 0.0)


if __name__ == "__main__":
    unittest.main()
