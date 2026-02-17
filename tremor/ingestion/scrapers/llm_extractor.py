"""LLM-based field extractor for unstructured event sources.

Adapted from smart-webscraper-products/src/extractors/llm_extractor.py.

Takes raw HTML, cleans it, sends it to Claude with a per-source JSON
schema, and returns a validated dict of extracted fields.

Uses claude-sonnet-4-6 at temperature=0.0 for deterministic extraction.
Requires env var: ANTHROPIC_API_KEY
"""

import json
import logging
import re
from typing import Any, Optional

from anthropic import Anthropic
from bs4 import BeautifulSoup

from tremor.config import settings

logger = logging.getLogger(__name__)

MAX_HTML_CHARS = 50_000

# System prompt shared by all sources
SYSTEM_PROMPT = """\
You are a financial data extraction assistant. Your job is to read news articles,
press releases, and regulatory announcements and extract structured information.

Rules:
- Return ONLY a valid JSON object matching the schema provided. No prose, no markdown.
- If a field cannot be determined from the text, set it to null.
- For numeric fields, return numbers only (no units, no % signs).
- For summary_text, write 2-3 factual sentences in the third person.
- Do not infer or extrapolate — only extract what is explicitly stated.
"""


class LLMExtractor:
    """Extract structured fields from HTML using Claude."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or getattr(settings, "ANTHROPIC_API_KEY", None)
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        self._client = Anthropic(api_key=key)

    def extract(self, html: str, schema: dict[str, Any], url: str = "") -> dict:
        """Extract fields from HTML using the provided schema as the target format.

        Args:
            html: Raw page HTML
            schema: Dict defining the expected output fields and their types/descriptions.
                    Example: {"summary_text": "str", "actual_rate": "float|null", ...}
            url: Source URL, included in prompt for context

        Returns:
            Dict with extracted fields. Missing fields set to None.
        """
        cleaned = self._clean_html(html)
        schema_str = json.dumps(schema, indent=2)

        user_content = (
            f"Source URL: {url}\n\n"
            f"Target schema (extract these fields):\n{schema_str}\n\n"
            f"Page content:\n{cleaned}"
        )

        try:
            message = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = message.content[0].text.strip()
            return self._parse_response(raw, schema)

        except Exception as e:
            logger.error(f"LLM extraction failed for {url}: {e}")
            return self._empty_result(schema)

    def _clean_html(self, html: str) -> str:
        """Strip scripts, styles, and nav noise. Return plain text."""
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "meta", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Truncate to fit context
        return text[:MAX_HTML_CHARS]

    def _parse_response(self, raw: str, schema: dict) -> dict:
        """Parse LLM JSON response and fill missing fields with None."""
        # Strip any accidental markdown code fences
        raw = re.sub(r"^```json?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Raw: {raw[:200]}")
            return self._empty_result(schema)

        # Ensure all schema keys are present
        for key in schema:
            if key not in result:
                result[key] = None
        return result

    def _empty_result(self, schema: dict) -> dict:
        """Return a dict with all schema keys set to None."""
        return {key: None for key in schema}
