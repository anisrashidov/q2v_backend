"""ClinicalTrialsPort implementation for the ClinicalTrials.gov v2 REST API.

Handles:
* Parameter mapping  (QueryParams snake_case → API dot-notation strings)
* Rate-limit retries with exponential back-off (HTTP 429 + Retry-After)
* Transient-error retries (5xx, timeouts)
* Pagination via nextPageToken
"""
from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp

from agent.types import QueryParams, Study, StudyData


class ClinicalTrialsAPI:
    """Adapter for the ClinicalTrials.gov v2 studies endpoint.

    Parameters
    ----------
    client:
        Pre-built ``aiohttp.ClientSession``.  The lifespan context manager
        should own the session lifecycle and pass it in here.
    base_url:
        Override the base URL (e.g. for testing against a mock server).
    max_retries:
        Maximum attempts before raising ``RuntimeError``.
    """

    _DEFAULT_BASE = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(
        self,
        client: Optional[aiohttp.ClientSession] = None,
        base_url: str = _DEFAULT_BASE,
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self._base_url = base_url
        self._max_retries = max_retries

    # ── Public interface ───────────────────────────────────────────────────────

    async def search(self, params: QueryParams) -> StudyData:
        """Fetch one page of studies."""
        raw = await self._fetch(self._build_params(params))
        return self._parse_response(raw)

    async def search_all_pages(
        self,
        params: QueryParams,
        max_pages: int = 5,
    ) -> StudyData:
        """Fetch up to *max_pages*, concatenating study lists across pages."""
        api_params = self._build_params(params)
        all_studies: list[Study] = []
        total_count = 0
        next_token: Optional[str] = None

        for _ in range(max_pages):
            if next_token:
                api_params = {**api_params, "pageToken": next_token}

            raw = await self._fetch(api_params)
            page = self._parse_response(raw)

            all_studies.extend(page.studies)
            total_count = page.total_count
            next_token = page.next_page_token

            if not next_token:
                break

        return StudyData(
            studies=all_studies,
            total_count=total_count,
            next_page_token=next_token,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    _PHASE_MAP: dict[str, str] = {
        "EARLY_PHASE1": "0",
        "PHASE1": "1",
        "PHASE2": "2",
        "PHASE3": "3",
        "PHASE4": "4",
        "NA": "na",
    }

    _STUDY_TYPE_MAP: dict[str, str] = {
        "INTERVENTIONAL": "int",
        "OBSERVATIONAL": "obs",
        "OBSERVATIONAL_REGISTRY": "obs",
        "EXPANDED_ACCESS": "exp",
    }

    @staticmethod
    def _build_params(params: QueryParams) -> dict[str, str]:
        """Map QueryParams → raw API query-string dict."""
        api: dict[str, str] = {
            "format": "json",
            "pageSize": str(params.page_size),
            "countTotal": "true",
        }
        if params.query_cond:
            api["query.cond"] = params.query_cond
        if params.query_term:
            api["query.term"] = params.query_term
        if params.query_intr:
            api["query.intr"] = params.query_intr
        if params.query_spons:
            api["query.spons"] = params.query_spons
        if params.query_locn:
            api["query.locn"] = params.query_locn
        if params.filter_overall_status:
            api["filter.overallStatus"] = ",".join(params.filter_overall_status)

        agg_filters: list[str] = []
        if params.filter_phase:
            nums = ",".join(
                ClinicalTrialsAPI._PHASE_MAP[p]
                for p in params.filter_phase
                if p in ClinicalTrialsAPI._PHASE_MAP
            )
            if nums:
                agg_filters.append(f"phase:{nums}")
        if params.filter_funder_type:
            agg_filters.append(f"funderType:{','.join(params.filter_funder_type)}")
        if params.filter_study_type:
            mapped = ",".join(
                ClinicalTrialsAPI._STUDY_TYPE_MAP.get(s, s.lower())
                for s in params.filter_study_type
            )
            if mapped:
                agg_filters.append(f"studyType:{mapped}")
        if params.filter_sex:
            agg_filters.append(f"sex:{params.filter_sex.lower()}")
        if params.filter_age_range:
            agg_filters.append(f"ageRange:{','.join(params.filter_age_range)}")
        if agg_filters:
            api["aggFilters"] = "|".join(agg_filters)

        # filter.advanced — date range and/or intervention type
        adv_parts: list[str] = []
        if params.filter_start_year or params.filter_end_year:
            start = f"{params.filter_start_year}-01-01" if params.filter_start_year else "MIN"
            end = f"{params.filter_end_year}-12-31" if params.filter_end_year else "MAX"
            adv_parts.append(f"AREA[StartDate]RANGE[{start},{end}]")
        if params.filter_intervention_type:
            adv_parts.append(f"AREA[InterventionType]{params.filter_intervention_type}")
        if adv_parts:
            api["filter.advanced"] = " AND ".join(adv_parts)

        if params.fields:
            api["fields"] = ",".join(params.fields)
        if params.sort:
            api["sort"] = params.sort
        return api

    async def _fetch(self, params: dict[str, str]) -> dict:
        """GET the studies endpoint with retry / back-off.

        Retry strategy
        ──────────────
        * HTTP 429 → honour ``Retry-After`` header (or back-off 2^attempt * 2 s)
        * HTTP 5xx → exponential back-off
        * Timeout → exponential back-off
        * Other 4xx → re-raise immediately (no point retrying a client error)
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                async with self._client.get(self._base_url, params=params) as resp:
                    if resp.status == 429:
                        wait = int(resp.headers.get("Retry-After", str(2 ** attempt * 2)))
                        await asyncio.sleep(wait)
                        last_exc = Exception(f"429 Too Many Requests (attempt {attempt + 1})")
                        continue

                    resp.raise_for_status()
                    return await resp.json()

            except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as exc:
                last_exc = exc
                await asyncio.sleep(2**attempt)

            except aiohttp.ClientResponseError as exc:
                if exc.status >= 500:
                    last_exc = exc
                    await asyncio.sleep(2**attempt)
                    continue
                raise

        raise RuntimeError(
            f"ClinicalTrials.gov API failed after {self._max_retries} attempts"
        ) from last_exc

    @staticmethod
    def _parse_response(raw: dict) -> StudyData:
        studies = [Study.from_api(s) for s in raw.get("studies", [])]
        return StudyData(
            studies=studies,
            total_count=raw.get("totalCount", len(studies)),
            next_page_token=raw.get("nextPageToken"),
        )
