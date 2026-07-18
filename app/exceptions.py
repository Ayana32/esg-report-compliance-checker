"""Domain exceptions for the ESG Compliance Checker API."""


class ComplianceServiceError(Exception):
    """Base class for expected compliance-service errors."""


class CompanyNotFoundError(ComplianceServiceError):
    """Raised when no report data exists for a requested company."""


class RequirementNotFoundError(ComplianceServiceError):
    """Raised when a standard or requirement cannot be found."""


class NoEvidenceFoundError(ComplianceServiceError):
    """Raised when retrieval returns no evidence for the request."""


class ExternalServiceError(ComplianceServiceError):
    """Raised when an external dependency such as an LLM is unavailable."""