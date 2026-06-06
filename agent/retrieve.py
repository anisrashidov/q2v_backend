"""Stage 2 — RETRIEVE.

Thin wrapper around :class:`ports.ClinicalTrialsPort` that fetches studies
across one or more pages and returns a unified :class:`models.StudyData`.

All HTTP concerns (retries, back-off, pagination) are handled by the adapter;
this module contains no I/O logic of its own.
"""
from __future__ import annotations

from adapters.clinicaltrials import ClinicalTrialsAPI
from agent.types import QueryParams, StudyData


async def retrieve_studies(
    params: QueryParams,
    ct_api: ClinicalTrialsAPI,
    max_pages: int = 5,
) -> StudyData:
    """Stage 2 — fetch studies matching *params*, following pagination.

    Parameters
    ----------
    params:
        Validated query parameters produced by :func:`agent.interpret.interpret_query`.
    ct_api:
        ClinicalTrialsAPI adapter.
    max_pages:
        Upper bound on pages to fetch; prevents runaway pagination for broad
        queries.

    Returns
    -------
    StudyData
        All studies from the fetched pages plus total count and an optional
        continuation token if more pages exist beyond *max_pages*.
    """
    return await ct_api.search_all_pages(params, max_pages=max_pages)
