import unittest

from projetcrag.rag import Chunk, LocalRetriever, extractive_answer


class LocalRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk("ShuriX Tech forme les jeunes à l'intelligence artificielle.", 1),
            Chunk("Malaikadrone Defender est un drone intelligent de sécurité.", 2),
        ]

    def test_search_returns_the_relevant_page(self) -> None:
        result = LocalRetriever(self.chunks).search("Quel est le drone de sécurité ?", k=1)
        self.assertEqual(result[0].page, 2)

    def test_offline_answer_keeps_citations(self) -> None:
        answer = extractive_answer("formation intelligence artificielle", self.chunks[:1])
        self.assertIn("Sources : p. 1", answer)


if __name__ == "__main__":
    unittest.main()
