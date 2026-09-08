"""Regression checks for the Markdown response contract."""

import unittest

from chatbot.citations import build_citations
from chatbot.llm import format_markdown_response


class MarkdownFormattingTests(unittest.TestCase):
    def test_normalizes_common_list_markers(self):
        response = format_markdown_response("• First\r\n2) Second  ")

        self.assertEqual(response, "- First\n2. Second")

    def test_preserves_fenced_code_blocks(self):
        response = format_markdown_response("```text\n• Keep this\n1) Also keep\n```")

        self.assertEqual(response, "```text\n• Keep this\n1) Also keep\n```")

    def test_sources_are_a_markdown_section(self):
        sources = build_citations([{"title": "Rice Guide"}])

        self.assertEqual(sources, "## Sources\n\n1. Rice Guide")


if __name__ == "__main__":
    unittest.main()
