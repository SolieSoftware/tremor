"""Base classes for all Tremor event ingesters.

Every ingester — whether an API client or a web scraper — produces
EventPayload objects that the normaliser converts into EventCreate records
for the Tremor API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EventPayload:
    """Raw output from any ingester, before normalisation.

    Structured API sources populate raw_data directly.
    Unstructured scrapers populate raw_data with whatever the LLM
    could extract, plus summary_text for the human-readable summary.
    Any fields the LLM could not determine are left as None and excluded
    from the final raw_data dict by the normaliser.
    """

    # Required fields (every ingester must supply these)
    event_type: str           # e.g. "fed_announcement", "economic_data"
    event_subtype: str        # e.g. "rate_decision", "cpi_release"
    timestamp: datetime       # When the event occurred
    description: str          # One-line human-readable description
    source_name: str          # e.g. "Federal Reserve", "BLS", "Reuters"
    source_url: str           # URL the data was fetched from

    # raw_data fields — populated as available, None means unknown
    actual_rate: Optional[float] = None
    expected_rate: Optional[float] = None
    actual_cpi: Optional[float] = None
    expected_cpi: Optional[float] = None
    actual_nfp: Optional[float] = None
    expected_nfp: Optional[float] = None
    actual_gdp: Optional[float] = None
    expected_gdp: Optional[float] = None
    actual_eps: Optional[float] = None
    expected_eps: Optional[float] = None
    vix_before: Optional[float] = None
    vix_after: Optional[float] = None
    spread_before: Optional[float] = None
    spread_after: Optional[float] = None
    yield_before: Optional[float] = None
    yield_after: Optional[float] = None
    summary_text: Optional[str] = None

    # Arbitrary extra fields the LLM extracted (tone, severity, etc.)
    # These go into raw_data under their own keys but are not used by
    # signal transforms — they are available for display and filtering.
    extra: dict = field(default_factory=dict)

    # Tags derived during ingestion (e.g. ["fomc", "rate_hold"])
    tags: list[str] = field(default_factory=list)

    def to_raw_data(self) -> dict:
        """Build the raw_data dict for EventCreate.

        Only includes fields that are not None, plus source metadata
        and any extras from LLM extraction.
        """
        canonical_fields = {
            "actual_rate": self.actual_rate,
            "expected_rate": self.expected_rate,
            "actual_cpi": self.actual_cpi,
            "expected_cpi": self.expected_cpi,
            "actual_nfp": self.actual_nfp,
            "expected_nfp": self.expected_nfp,
            "actual_gdp": self.actual_gdp,
            "expected_gdp": self.expected_gdp,
            "actual_eps": self.actual_eps,
            "expected_eps": self.expected_eps,
            "vix_before": self.vix_before,
            "vix_after": self.vix_after,
            "spread_before": self.spread_before,
            "spread_after": self.spread_after,
            "yield_before": self.yield_before,
            "yield_after": self.yield_after,
            "summary_text": self.summary_text,
            "source_url": self.source_url,
            "source_name": self.source_name,
        }
        result = {k: v for k, v in canonical_fields.items() if v is not None}
        result.update(self.extra)
        return result


class BaseIngester(ABC):
    """Abstract base class for all event ingesters.

    Subclasses implement fetch() to return a list of EventPayload objects.
    The caller is responsible for normalising these into EventCreate and
    posting them to the Tremor API (or writing directly to the DB).
    """

    @abstractmethod
    async def fetch(self, **kwargs) -> list[EventPayload]:
        """Fetch events from the source and return as EventPayload list."""
        ...

    def source_name(self) -> str:
        return self.__class__.__name__
