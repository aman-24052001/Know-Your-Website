from pydantic import BaseModel, HttpUrl


class AuditRequest(BaseModel):
    url: str
