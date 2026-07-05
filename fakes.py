"""
fakes.py

Shared test doubles for the Confluence HTTP layer, used by both
test_confluence_client.py (unit tests) and test_pipeline_integration.py
(end-to-end pipeline test) so the mock behavior only has to be right in
one place.
"""

SAMPLE_STORAGE_HTML = (
    "<h1>Data Sharing Overview</h1>"
    "<p>This customer requests anonymized vehicle trajectory data.</p>"
    "<ul><li>Format: CSV</li><li>Frequency: daily batch</li></ul>"
)


class FakeResponse:
    def __init__(self, status_code: int, json_data: dict, headers: dict = None):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """
    Minimal stand-in for requests.Session. `responses` is a list of
    FakeResponse objects returned in order, one per call to .get().
    """

    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.headers: dict = {}
        self.auth = None
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params or {}))
        return self._responses.pop(0)


def make_confluence_page_flow(title: str, storage_html: str, webui_path: str, version: int = 1):
    """
    Build the 3-response sequence a ConfluenceClient.get_customer_docs()
    call makes: space lookup -> page list -> page detail.
    """
    return [
        FakeResponse(200, {"results": [{"id": "space-1"}]}),
        FakeResponse(200, {"results": [{"id": "page-1"}]}),
        FakeResponse(200, {
            "title": title,
            "version": {"number": version},
            "_links": {"webui": webui_path},
            "body": {"storage": {"value": storage_html}},
        }),
    ]
