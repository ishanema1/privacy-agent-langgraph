"""
confluence_client.py

Implements step 1 of the pipeline: "Context gathering — Confluence API."

Each customer's architecture docs, data specs, and use-case description are
handed over once and stored in a dedicated internal Confluence space. This
module fetches and normalizes that content into plain text so it can be fed
into the retrieval/drafting steps downstream.

Design notes:
  - Read-only. This client only ever GETs content — it has no methods that
    create, edit, or delete Confluence content, since the pipeline's job is
    to consume customer docs, not modify the source of truth.
  - Auth via environment variables, never hardcoded. Uses HTTP Basic Auth
    with an email + API token, which is how Atlassian Cloud's REST API
    expects credentials (https://developer.atlassian.com/cloud/confluence/).
  - Confluence stores page bodies as "storage format" (XHTML-ish markup).
    `_extract_text` strips that down to plain text suitable for an LLM
    prompt — headings and paragraph breaks are preserved, markup isn't.
  - Handles pagination and basic 429 backoff, since both are the most
    common real-world gotchas when pulling more than a handful of pages.

This is a sanitized, standalone reference implementation for a public
portfolio case study. No real space keys, credentials, or customer content
are used or represented here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import requests
from bs4 import BeautifulSoup

DEFAULT_PAGE_SIZE = 25
MAX_RETRIES = 3


class ConfluenceAuthError(RuntimeError):
    """Raised when the API rejects our credentials."""


class ConfluenceClient:
    """
    Thin, read-only wrapper around the Confluence Cloud REST API for
    pulling a customer's docs out of their dedicated space.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = (base_url or os.environ["CONFLUENCE_BASE_URL"]).rstrip("/")
        email = email or os.environ["CONFLUENCE_EMAIL"]
        api_token = api_token or os.environ["CONFLUENCE_API_TOKEN"]

        self._session = session or requests.Session()
        self._session.auth = (email, api_token)
        self._session.headers.update({"Accept": "application/json"})

    # ----------------------------------------------------------------
    # Low-level request handling
    # ----------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"

        for attempt in range(1, MAX_RETRIES + 1):
            response = self._session.get(url, params=params, timeout=15)

            if response.status_code == 401:
                raise ConfluenceAuthError(
                    "Confluence rejected our credentials (401). "
                    "Check CONFLUENCE_EMAIL / CONFLUENCE_API_TOKEN."
                )

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            return response.json()

        raise RuntimeError(f"Confluence API rate-limited us after {MAX_RETRIES} retries: {url}")

    def _get_paginated(self, path: str, params: Optional[dict] = None) -> Iterator[dict]:
        """Yield every result across all pages for a Confluence list endpoint."""
        params = dict(params or {})
        params.setdefault("limit", DEFAULT_PAGE_SIZE)
        start = 0

        while True:
            params["start"] = start
            payload = self._get(path, params=params)
            results = payload.get("results", [])
            yield from results

            if len(results) < params["limit"]:
                return
            start += params["limit"]

    # ----------------------------------------------------------------
    # Content normalization
    # ----------------------------------------------------------------

    @staticmethod
    def _extract_text(storage_format_html: str) -> str:
        """
        Convert Confluence's storage-format markup into plain text suitable
        for an LLM prompt: headings and paragraphs are kept on their own
        lines, all markup is discarded.
        """
        soup = BeautifulSoup(storage_format_html or "", "html.parser")

        block_tags = ("h1", "h2", "h3", "h4", "p", "li", "tr")
        for tag in soup.find_all(block_tags):
            tag.append("\n")

        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def get_customer_docs(self, space_key: str) -> list["CustomerDoc"]:
        """
        Fetch every page in a customer's dedicated Confluence space and
        return normalized, plain-text documents ready for the retrieval
        and drafting steps.
        """
        docs = []
        for page_summary in self._get_paginated(
            "/wiki/api/v2/pages",
            params={"space-id": self._resolve_space_id(space_key)},
        ):
            docs.append(self._to_customer_doc(page_summary["id"]))
        return docs

    def _resolve_space_id(self, space_key: str) -> str:
        payload = self._get("/wiki/api/v2/spaces", params={"keys": space_key})
        results = payload.get("results", [])
        if not results:
            raise ValueError(f"No Confluence space found for key '{space_key}'")
        return results[0]["id"]

    def _to_customer_doc(self, page_id: str) -> "CustomerDoc":
        payload = self._get(
            f"/wiki/api/v2/pages/{page_id}",
            params={"body-format": "storage"},
        )
        raw_html = payload["body"]["storage"]["value"]
        return CustomerDoc(
            page_id=page_id,
            title=payload["title"],
            content=self._extract_text(raw_html),
            version=payload["version"]["number"],
            url=f"{self.base_url}/wiki{payload['_links']['webui']}",
        )


@dataclass
class CustomerDoc:
    """A single normalized Confluence page, ready to feed into the pipeline."""

    page_id: str
    title: str
    content: str
    version: int
    url: str

    def as_context_block(self) -> str:
        """Format this doc for inclusion in an LLM prompt's context window."""
        return f"### {self.title}\n(source: {self.url})\n\n{self.content}"


def _demo() -> None:
    """
    Illustrative usage — not runnable without real credentials and a real
    space. See test_confluence_client.py for a fully offline, mocked
    version of this same flow.
    """
    client = ConfluenceClient()  # reads CONFLUENCE_BASE_URL / EMAIL / API_TOKEN from env
    docs = client.get_customer_docs(space_key="CUST014")  # anonymized example space key
    for doc in docs:
        print(doc.as_context_block())
        print("-" * 60)


if __name__ == "__main__":
    _demo()
