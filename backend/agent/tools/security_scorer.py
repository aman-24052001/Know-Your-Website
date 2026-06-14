from backend.models.report import ModuleResult, SeverityLevel


DEDUCTIONS = {
    SeverityLevel.CRITICAL: 25,
    SeverityLevel.HIGH: 15,
    SeverityLevel.MEDIUM: 8,
    SeverityLevel.LOW: 3,
}


def compute_score(modules: list[ModuleResult]) -> int:
    score = 100
    for module in modules:
        for finding in module.findings:
            if finding.severity in DEDUCTIONS:
                score -= DEDUCTIONS[finding.severity]
    return max(0, min(100, score))


def module_status_label(status: SeverityLevel) -> str:
    labels = {
        SeverityLevel.PASS: "PASS",
        SeverityLevel.LOW: "LOW",
        SeverityLevel.MEDIUM: "MEDIUM",
        SeverityLevel.HIGH: "HIGH",
        SeverityLevel.CRITICAL: "CRITICAL",
    }
    return labels.get(status, "UNKNOWN")
