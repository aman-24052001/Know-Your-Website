from pydantic import BaseModel
from enum import Enum
from typing import Optional


class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    PASS = "PASS"


class Finding(BaseModel):
    key: str
    detail: str
    severity: SeverityLevel


class ModuleResult(BaseModel):
    module: str
    status: SeverityLevel  # worst severity in this module
    findings: list[Finding]
    raw: Optional[dict] = None


class AuditReport(BaseModel):
    url: str
    score: int
    modules: list[ModuleResult]
    summary: str
    scanned_at: str
    checks_performed: int
