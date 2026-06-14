from typing import TypedDict, AsyncGenerator, Optional
from backend.models.report import ModuleResult


class AuditState(TypedDict):
    url: str
    module_results: list[ModuleResult]
    log_lines: list[str]
    final_report: Optional[dict]
