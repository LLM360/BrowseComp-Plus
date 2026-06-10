"""
Remote searcher proxy for an HTTP retrieval service.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from urllib import error, request

from .base import BaseSearcher

logger = logging.getLogger(__name__)


class RemoteSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--retriever-url",
            required=True,
            help="Base URL for a retrieval server exposing /retrieve.",
        )
        parser.add_argument(
            "--retriever-timeout",
            type=float,
            default=120.0,
            help="Timeout in seconds for remote retrieval requests.",
        )

    def __init__(self, args):
        self.args = args
        self.base_url = args.retriever_url.rstrip("/")
        self.timeout = args.retriever_timeout
        logger.info("Using remote retriever at %s", self.base_url)

    def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Remote retriever returned HTTP {exc.code} for {path}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Failed to reach remote retriever at {self.base_url}{path}: {exc}"
            ) from exc

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        response = self._post_json("/retrieve", {"query": query})
        raw_results = response.get("result", response)
        if not isinstance(raw_results, list):
            raise RuntimeError(f"Unexpected remote retriever response: {response!r}")

        results: list[dict[str, Any]] = []
        for item in raw_results[:k]:
            if not isinstance(item, dict):
                continue

            docid = item.get("docid")
            score = item.get("score")
            text = item.get("text")

            document = item.get("document")
            if isinstance(document, dict):
                title = str(document.get("title") or "").strip()
                body = str(document.get("text") or "").strip()
                text = f"{title}\n{body}".strip() if title else body

            if docid is None or text is None:
                continue

            results.append({"docid": str(docid), "score": score, "text": str(text)})

        return results

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        try:
            response = self._post_json("/document", {"docid": docid})
        except RuntimeError as exc:
            logger.warning("Remote document lookup failed for %s: %s", docid, exc)
            return None

        result = response.get("result", response)
        if not result:
            return None
        if not isinstance(result, dict):
            return None
        if "text" not in result:
            return None
        return {"docid": str(result.get("docid", docid)), "text": str(result["text"])}

    @property
    def search_type(self) -> str:
        return "remote"
