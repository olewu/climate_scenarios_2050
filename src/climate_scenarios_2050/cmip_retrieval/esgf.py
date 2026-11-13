"""
ESGF search over several index nodes.

Since 2025 the federation is split: the European index nodes (DKRZ, CEDA) and the US bridge (ORNL,
ESGF-NG) list different data nodes, so a complete inventory needs the union of all of them. Every
index node is queried with `distrib=true`; a node that fails is logged and skipped.
"""

import logging
import time

import requests

log = logging.getLogger(__name__)

# url, page size, whether the node accepts the `fields` parameter
INDEX_NODES = {
    "dkrz": {"url": "https://esgf-data.dkrz.de/esg-search/search", "page": 1000, "fields": True},
    "ceda": {"url": "https://esgf.ceda.ac.uk/esg-search/search", "page": 1000, "fields": True},
    "ornl": {"url": "https://esgf-node.ornl.gov/esgf-1-5-bridge", "page": 500, "fields": False},
}

DATASET_FIELDS = [
    "instance_id", "source_id", "variant_label", "grid_label", "version", "size", "number_of_files",
    "data_node", "experiment_id", "variable_id", "table_id", "retracted", "replica",
]
FILE_FIELDS = ["instance_id", "dataset_id", "title", "size", "checksum", "checksum_type", "url", "data_node", "retracted"]


def _get(session, url, params, timeout, retries=4):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()["response"]
        except Exception as exc:  # network errors, server errors, non-json answers
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            log.warning(f"{url}: {exc!r}, retrying in {wait} s")
            time.sleep(wait)


def search(node: str, kind: str = "Dataset", timeout: float = 300, **facets) -> list:
    """all documents of one index node for the given facets (paginated)"""
    spec = INDEX_NODES[node]
    params = {"project": "CMIP6", "type": kind, "latest": "true", "distrib": "true", "format": "application/solr+json", "limit": spec["page"], **facets}
    if spec["fields"]:
        params["fields"] = ",".join(DATASET_FIELDS if kind == "Dataset" else FILE_FIELDS)
    docs, offset, found = [], 0, None
    with requests.Session() as session:
        while found is None or offset < found:
            resp = _get(session, spec["url"], {**params, "offset": offset}, timeout)
            found = int(resp["numFound"])
            batch = resp["docs"]
            if not batch:
                break
            docs.extend(batch)
            offset += len(batch)
    return docs


def search_all(nodes=None, kind="Dataset", **facets) -> tuple:
    """(documents of all reachable index nodes, each tagged with `index_node`; {node: error} of failed nodes)"""
    docs, failed = [], {}
    for node in nodes or INDEX_NODES:
        try:
            found = search(node, kind, **facets)
        except Exception as exc:
            log.error(f"index node {node} failed for {facets}: {exc!r}")
            failed[node] = repr(exc)
            continue
        for doc in found:
            doc["index_node"] = node
        docs.extend(found)
        log.info(f"{node}: {len(found)} {kind.lower()} records for {facets}")
    return docs, failed


def first(value):
    """ESGF returns many scalars as one-element lists"""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def http_urls(doc) -> list:
    return [u.split("|")[0] for u in doc.get("url") or [] if u.endswith("|HTTPServer")]


def opendap_urls(doc) -> list:
    return [u.split("|")[0].removesuffix(".html") for u in doc.get("url") or [] if u.endswith("|OPENDAP")]
