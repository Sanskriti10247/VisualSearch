"""Request/response schemas (also the source of the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Orientation = Literal["landscape", "portrait", "square"]
Reranker = Literal["none", "prf", "visual_prf", "cross_encoder"]
Fusion = Literal["zscore", "minmax", "rrf", "early"]


class Filters(BaseModel):
    """Metadata filters. Values within a field are OR-ed; fields are AND-ed."""

    model_config = ConfigDict(extra="forbid")

    collections: list[str] = Field(default_factory=list, max_length=50)
    splits: list[str] = Field(default_factory=list, max_length=10)
    labels: list[str] = Field(default_factory=list, max_length=100)
    tags_any: list[str] = Field(default_factory=list, max_length=50)
    orientations: list[Orientation] = Field(default_factory=list)
    min_width: int | None = Field(default=None, ge=1, le=100_000)
    min_height: int | None = Field(default=None, ge=1, le=100_000)
    exclude_ids: list[int] = Field(default_factory=list, max_length=1000)


class TextSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"query": "a red backpack placed on a wooden table near a window", "k": 12}]})

    query: str = Field(min_length=1, max_length=500)
    k: int | None = Field(default=None, ge=1, le=200)
    model: str | None = Field(default=None, description="text-capable model key (CLIP)")
    filters: Filters | None = None
    rerank: Reranker = "none"
    candidates: int | None = Field(default=None, ge=1, le=1000,
                                   description="first-stage candidates when reranking")
    min_prob: float | None = Field(default=None, ge=0, le=1,
                                   description="drop results whose calibrated P(match) is lower")

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class SignalScore(BaseModel):
    raw: float = Field(description="cosine similarity in this embedding space")
    z: float | None = Field(default=None, description="standardised vs. random image pairs (val)")
    percentile: float | None = Field(default=None, description="fraction of random pairs scoring lower")
    prob: float | None = Field(default=None, description="calibrated P(annotated match)")


class ImageMeta(BaseModel):
    id: int
    collection: str
    split: str | None = None
    source_name: str | None = None
    width: int
    height: int
    orientation: str
    file_size: int
    format: str | None = None
    label: str | None = None
    tags: list[str] = Field(default_factory=list)
    sha256: str
    created_at: str | None = None
    extra: dict[str, Any] | None = None


class SearchHit(BaseModel):
    rank: int
    image_id: int
    score: float = Field(description="final ranking score (scale depends on strategy)")
    signals: dict[str, SignalScore]
    thumbnail_url: str
    image_url: str
    metadata: ImageMeta


class SearchResponse(BaseModel):
    request_id: str
    results: list[SearchHit]
    total: int
    k: int
    timings_ms: dict[str, float]
    strategy: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class ImageDetail(BaseModel):
    metadata: ImageMeta
    captions: list[str]
    thumbnail_url: str
    image_url: str
    indexed_in: list[str]


class JobInfo(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    updated_at: str
    progress: int
    total: int
    result: dict[str, Any] | None = None
    error: str | None = None


class IndexJobAccepted(BaseModel):
    job_id: str
    status_url: str
    accepted_files: int


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
