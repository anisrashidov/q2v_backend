"""ClinicalTrialsPort implementation for the ClinicalTrials.gov v2 REST API.

Handles:
* Parameter mapping  (QueryParams snake_case → API dot-notation strings)
* Rate-limit retries with exponential back-off (HTTP 429 + Retry-After)
* Transient-error retries (5xx, timeouts)
* Pagination via nextPageToken
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

from agent.types import QueryParams, Study, StudyData

logger = logging.getLogger(__name__)


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

    # Essie equivalents for aggFilter values, used when filter.advanced is also
    # present (the CT.gov v2 API rejects requests that combine the two params).
    _AGE_RANGE_ESSIE: dict[str, str] = {
        "child": "AREA[StdAge]Child",
        "adult": "AREA[StdAge]Adult",
        "older": 'AREA[StdAge]"Older Adult"',
    }
    _FUNDER_TYPE_ESSIE: dict[str, str] = {
        "NIH": "AREA[LeadSponsorClass]NIH",
        "OTHER_GOV": "AREA[LeadSponsorClass]OTHER_GOV",
        "INDIV": "AREA[LeadSponsorClass]INDIV",
        "INDUSTRY": "AREA[LeadSponsorClass]INDUSTRY",
        "OTHER": "AREA[LeadSponsorClass]OTHER",
        "FED": "AREA[LeadSponsorClass]FED",
        "NETWORK": "AREA[LeadSponsorClass]NETWORK",
        "UNKNOWN": "AREA[LeadSponsorClass]UNKNOWN",
    }

    @staticmethod
    def _essie_or(parts: list[str]) -> str:
        """Wrap multiple Essie terms in parens with OR, or return the single term."""
        return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"

    @staticmethod
    def _build_params(params: QueryParams) -> dict[str, str]:
        """Map QueryParams → raw API query-string dict.

        The CT.gov v2 API does not allow aggFilters and filter.advanced in the
        same request.  We build filter.advanced first; if it will be non-empty,
        every aggFilter is converted to an equivalent Essie expression and merged
        into filter.advanced instead of being sent as aggFilters.
        """
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

        # Build filter.advanced parts first so we know whether aggFilters must
        # be promoted to Essie expressions.
        adv_parts: list[str] = []
        if params.filter_start_year or params.filter_end_year:
            start = f"{params.filter_start_year}-01-01" if params.filter_start_year else "MIN"
            end = f"{params.filter_end_year}-12-31" if params.filter_end_year else "MAX"
            adv_parts.append(f"AREA[StartDate]RANGE[{start},{end}]")
        if params.filter_intervention_type:
            adv_parts.append(f"AREA[InterventionType]{params.filter_intervention_type}")

        # When filter.advanced is in use, all categorical filters must go there
        # too (as Essie). When it is not in use, the simpler aggFilters suffices.
        use_advanced = bool(adv_parts)
        agg_filters: list[str] = []

        if params.filter_phase:
            if use_advanced:
                parts = [f"AREA[Phase]{p}" for p in params.filter_phase]
                adv_parts.append(ClinicalTrialsAPI._essie_or(parts))
            else:
                nums = ",".join(
                    ClinicalTrialsAPI._PHASE_MAP[p]
                    for p in params.filter_phase
                    if p in ClinicalTrialsAPI._PHASE_MAP
                )
                if nums:
                    agg_filters.append(f"phase:{nums}")

        if params.filter_funder_type:
            if use_advanced:
                parts = [
                    ClinicalTrialsAPI._FUNDER_TYPE_ESSIE.get(ft, f"AREA[LeadSponsorClass]{ft}")
                    for ft in params.filter_funder_type
                ]
                adv_parts.append(ClinicalTrialsAPI._essie_or(parts))
            else:
                agg_filters.append(f"funderType:{','.join(params.filter_funder_type)}")

        if params.filter_study_type:
            if use_advanced:
                parts = [f"AREA[StudyType]{s}" for s in params.filter_study_type]
                adv_parts.append(ClinicalTrialsAPI._essie_or(parts))
            else:
                mapped = ",".join(
                    ClinicalTrialsAPI._STUDY_TYPE_MAP.get(s, s.lower())
                    for s in params.filter_study_type
                )
                if mapped:
                    agg_filters.append(f"studyType:{mapped}")

        if params.filter_sex:
            sex_val = params.filter_sex.lower()
            if use_advanced:
                if sex_val != "all":
                    adv_parts.append(f"AREA[Sex]{sex_val.title()}")
            else:
                agg_filters.append(f"sex:{sex_val}")

        if params.filter_age_range:
            if use_advanced:
                parts = [
                    ClinicalTrialsAPI._AGE_RANGE_ESSIE[a]
                    for a in params.filter_age_range
                    if a in ClinicalTrialsAPI._AGE_RANGE_ESSIE
                ]
                if parts:
                    adv_parts.append(ClinicalTrialsAPI._essie_or(parts))
            else:
                agg_filters.append(f"ageRange:{','.join(params.filter_age_range)}")

        if agg_filters:
            api["aggFilters"] = "|".join(agg_filters)
        if adv_parts:
            api["filter.advanced"] = " AND ".join(adv_parts)

        if params.fields:
            api["fields"] = ",".join(params.fields)
        if params.sort:
            api["sort"] = params.sort
        return api

    async def fetch_by_nct_ids(self, nct_ids: list[str]) -> StudyData:
        """Fetch specific studies by their NCT IDs."""
        api_params = {
            "format": "json",
            "filter.ids": ",".join(nct_ids),
            "pageSize": str(min(len(nct_ids), 1000)),
            "countTotal": "true",
        }
        raw = await self._fetch(api_params)
        return self._parse_response(raw)

    async def fetch_study_details(self, nct_id: str) -> dict:
        """Fetch full protocol detail for a single study.

        Returns a curated dict with eligibility, outcomes, locations, and arms
        in addition to the standard fields available via search.
        """
        url = f"{self._base_url}/{nct_id}"
        raw = await self._fetch({}, url=url)
        proto = raw.get("protocolSection", {})
        return {
            "nct_id": nct_id,
            "title": proto.get("identificationModule", {}).get("briefTitle"),
            "official_title": proto.get("identificationModule", {}).get("officialTitle"),
            "status": proto.get("statusModule", {}).get("overallStatus"),
            "phases": proto.get("designModule", {}).get("phases", []),
            "study_type": proto.get("designModule", {}).get("studyType"),
            "conditions": proto.get("conditionsModule", {}).get("conditions", []),
            "keywords": proto.get("conditionsModule", {}).get("keywords", []),
            "sponsor": proto.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
            "enrollment": proto.get("designModule", {}).get("enrollmentInfo", {}).get("count"),
            "eligibility": proto.get("eligibilityModule", {}).get("eligibilityCriteria"),
            "min_age": proto.get("eligibilityModule", {}).get("minimumAge"),
            "max_age": proto.get("eligibilityModule", {}).get("maximumAge"),
            "sex": proto.get("eligibilityModule", {}).get("sex"),
            "primary_outcomes": [
                o.get("measure") for o in
                proto.get("outcomesModule", {}).get("primaryOutcomes", [])
            ],
            "secondary_outcomes": [
                o.get("measure") for o in
                proto.get("outcomesModule", {}).get("secondaryOutcomes", [])
            ],
            "arms": [
                {"label": a.get("label"), "type": a.get("type"), "description": a.get("description")}
                for a in proto.get("armsInterventionsModule", {}).get("armGroups", [])
            ],
            "locations_count": len(
                proto.get("contactsLocationsModule", {}).get("locations", [])
            ),
            "countries": list({
                loc.get("country", "")
                for loc in proto.get("contactsLocationsModule", {}).get("locations", [])
                if loc.get("country")
            }),
        }

    async def _fetch(self, params: dict[str, str], url: Optional[str] = None) -> dict:
        """GET the studies endpoint with retry / back-off.

        Retry strategy
        ──────────────
        * HTTP 429 → honour ``Retry-After`` header (or back-off 2^attempt * 2 s)
        * HTTP 5xx → exponential back-off
        * Timeout → exponential back-off
        * Other 4xx → re-raise immediately (no point retrying a client error)
        """
        target = url or self._base_url
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                logger.debug(
                    "ClinicalTrials GET %s params=%s (attempt %d/%d)",
                    target,
                    params,
                    attempt + 1,
                    self._max_retries,
                )
                async with self._client.get(target, params=params) as resp:
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
