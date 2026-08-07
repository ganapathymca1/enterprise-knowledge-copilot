"""Tool/function layer: structured HR record lookups with access control."""

from .hr_records import AccessDenied, HRRecords, load_records
from .registry import (
    POLICY_SEARCH,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    route_deterministically,
)

__all__ = [
    "AccessDenied",
    "HRRecords",
    "POLICY_SEARCH",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "load_records",
    "route_deterministically",
]
