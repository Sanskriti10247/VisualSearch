"""FastAPI application.

Handlers only parse/validate input, call the engine or services, and shape
responses. Inference, indexing and ranking live in the library modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from visualsearch.api.schemas import (
    ErrorResponse,
    Filters,
    ImageDetail,
    ImageMeta,
    IndexJobAccepted,
    JobInfo,
    SearchHit,
    SearchResponse,
    TextSearchRequest,
)
from visualsearch.api.services import Services
from visualsearch.config import Settings, get_settings
from visualsearch.data.catalog import FilterSpec
from visualsearch.data.preprocess import DecodedImage, InvalidImageError, decode_image
from visualsearch.embeddings import UnsupportedModalityError
from visualsearch.retrieval.engine import SearchError, SearchOutcome

log = logging.getLogger(__name__)
API = "/api/v1"


# ------------------------------------------------------------------ errors
class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Any = None):
        self.status, self.code, self.message, self.details = status, code, message, details


def _error(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "error": {"code": code, "message": message, "details": details}})


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api(_: Request, e: APIError):
        return _error(e.status, e.code, e.message, e.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        details = [{"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
                   for err in e.errors()]
        return _error(422, "validation_error", "Request validation failed", details)

    @app.exception_handler(SearchError)
    async def _search(_: Request, e: SearchError):
        return _error(422, "invalid_search", str(e))

    @app.exception_handler(UnsupportedModalityError)
    async def _modality(_: Request, e: UnsupportedModalityError):
        return _error(422, "unsupported_modality", str(e))

    @app.exception_handler(InvalidImageError)
    async def _image(_: Request, e: InvalidImageError):
        return _error(415, "invalid_image", str(e))

    @app.exception_handler(HTTPException)
    async def _http(_: Request, e: HTTPException):
        return _error(e.status_code, "http_error", str(e.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception):
        log.exception("unhandled error")
        return _error(500, "internal_error", "Internal server error")


# ------------------------------------------------------------- dependencies
def services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(services)]


def writable(request: Request) -> None:
    """Routes that change the collection (or report on such changes) are off in read-only mode."""
    if services(request).settings.read_only:
        raise APIError(403, "read_only",
                       "this server is read-only: indexing, deletion and jobs are disabled")


async def read_upload(file: UploadFile, settings: Settings) -> bytes:
    """Read an upload with a hard size cap (never buffers more than cap+1 MB)."""
    limit = int(settings.max_upload_mb * 1024 * 1024)
    chunks, size = [], 0
    while chunk := await file.read(1 << 20):
        size += len(chunk)
        if size > limit:
            raise APIError(413, "file_too_large",
                           f"'{file.filename}' exceeds the {settings.max_upload_mb:g} MB limit")
        chunks.append(chunk)
    if size == 0:
        raise APIError(400, "empty_file", f"'{file.filename}' is empty")
    return b"".join(chunks)


async def decode_upload(file: UploadFile, settings: Settings) -> DecodedImage:
    data = await read_upload(file, settings)
    return await run_in_threadpool(decode_image, data, settings.max_image_pixels)


def parse_filters(raw: str | None) -> FilterSpec | None:
    if not raw:
        return None
    try:
        f = Filters.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        raise APIError(422, "invalid_filters", "filters must be a JSON object matching Filters",
                       str(e)) from e
    return FilterSpec.from_dict(f.model_dump())


def to_filter_spec(f: Filters | None) -> FilterSpec | None:
    return FilterSpec.from_dict(f.model_dump()) if f else None


def build_response(svc: Services, outcome: SearchOutcome, k: int, request_id: str
                   ) -> SearchResponse:
    t = time.perf_counter()
    recs = svc.catalog.get_many([h.image_id for h in outcome.hits])
    results = []
    for rank, h in enumerate(outcome.hits, start=1):
        rec = recs.get(h.image_id)
        if rec is None or rec.deleted:
            continue
        results.append(SearchHit(
            rank=rank, image_id=h.image_id, score=h.score, signals=h.signals,
            thumbnail_url=f"{API}/images/{h.image_id}/thumbnail",
            image_url=f"{API}/images/{h.image_id}/file",
            metadata=ImageMeta(**rec.to_public())))
    timings = dict(outcome.timings_ms)
    timings["hydrate"] = round(1000 * (time.perf_counter() - t), 3)
    svc.metrics.observe_stages(outcome.strategy.get("mode", "search"), timings)
    return SearchResponse(request_id=request_id, results=results, total=len(results), k=k,
                          timings_ms=timings, strategy=outcome.strategy, warnings=outcome.warnings)


# -------------------------------------------------------------------- routes
router = APIRouter(prefix=API)
ERRORS = {code: {"model": ErrorResponse} for code in (400, 404, 413, 415, 422, 500)}
WRITE_ERRORS = {**ERRORS, 403: {"model": ErrorResponse}}


@router.post("/search/text", response_model=SearchResponse, responses=ERRORS, tags=["search"])
async def search_text(req: TextSearchRequest, svc: Svc) -> SearchResponse:
    """Natural-language query -> images (CLIP text encoder vs. CLIP image index)."""
    k = svc.settings.default_k if req.k is None else req.k
    out = await run_in_threadpool(
        svc.engine.text, req.query, k, req.model, to_filter_spec(req.filters), req.rerank,
        req.candidates, req.min_prob)
    return build_response(svc, out, k, uuid.uuid4().hex[:12])


@router.post("/search/image", response_model=SearchResponse, responses=ERRORS, tags=["search"])
async def search_image(
    svc: Svc,
    file: Annotated[UploadFile | None, File(description="query image")] = None,
    image_id: Annotated[int | None, Form(description="use an indexed image as the query")] = None,
    k: Annotated[int | None, Form(ge=1, le=200)] = None,
    model: Annotated[str | None, Form()] = None,
    filters: Annotated[str | None, Form(description="JSON-encoded Filters")] = None,
    rerank: Annotated[str, Form(pattern="^(none|prf)$")] = "none",
    exclude_query: Annotated[bool, Form()] = True,
    min_prob: Annotated[float | None, Form(ge=0, le=1)] = None,
) -> SearchResponse:
    """Query by example image (upload a file, or reference an indexed ``image_id``)."""
    if (file is None) == (image_id is None):
        raise APIError(422, "invalid_search", "provide exactly one of 'file' or 'image_id'")
    dec = await decode_upload(file, svc.settings) if file is not None else None
    k = svc.settings.default_k if k is None else k
    out = await run_in_threadpool(
        svc.engine.image, dec.image if dec else None, image_id, k, model, parse_filters(filters),
        rerank, exclude_query, None, min_prob, dec.sha256 if dec else None)
    return build_response(svc, out, k, uuid.uuid4().hex[:12])


@router.post("/search/hybrid", response_model=SearchResponse, responses=ERRORS, tags=["search"])
async def search_hybrid(
    svc: Svc,
    text: Annotated[str | None, Form(max_length=500)] = None,
    file: Annotated[UploadFile | None, File()] = None,
    image_id: Annotated[int | None, Form()] = None,
    k: Annotated[int | None, Form(ge=1, le=200)] = None,
    text_model: Annotated[str | None, Form()] = None,
    image_model: Annotated[str | None, Form()] = None,
    text_weight: Annotated[float, Form(ge=0, le=1)] = 0.5,
    fusion: Annotated[str, Form(pattern="^(zscore|minmax|rrf|early)$")] = "zscore",
    filters: Annotated[str | None, Form(description="JSON-encoded Filters")] = None,
    exclude_query: Annotated[bool, Form()] = True,
    min_prob: Annotated[float | None, Form(ge=0, le=1)] = None,
) -> SearchResponse:
    """Text + example image (+ metadata filters), fused at the score level."""
    if file is not None and image_id is not None:
        raise APIError(422, "invalid_search", "provide at most one of 'file' or 'image_id'")
    dec = await decode_upload(file, svc.settings) if file is not None else None
    k = svc.settings.default_k if k is None else k
    out = await run_in_threadpool(
        svc.engine.hybrid, text, dec.image if dec else None, image_id, k, text_model, image_model,
        text_weight, fusion, parse_filters(filters), None, exclude_query, min_prob,
        dec.sha256 if dec else None)
    return build_response(svc, out, k, uuid.uuid4().hex[:12])


@router.post("/index/images", response_model=IndexJobAccepted, status_code=202,
             responses=WRITE_ERRORS, tags=["index"], dependencies=[Depends(writable)])
async def index_images(
    svc: Svc,
    files: Annotated[list[UploadFile], File(description="images to ingest")],
    collection: Annotated[str, Form(min_length=1, max_length=64, pattern=r"^[\w\-]+$")] = "uploads",
    tags: Annotated[str | None, Form(description="comma-separated tags")] = None,
) -> IndexJobAccepted:
    """Validate uploads, then ingest + embed + index them in a background job."""
    if len(files) > svc.settings.max_batch_files:
        raise APIError(413, "too_many_files",
                       f"at most {svc.settings.max_batch_files} files per request")
    payload, rejected = [], []
    for f in files:
        data = await read_upload(f, svc.settings)
        try:
            await run_in_threadpool(decode_image, data, svc.settings.max_image_pixels)
        except InvalidImageError as e:
            rejected.append({"file": f.filename, "error": str(e)})
            continue
        payload.append((f.filename or "upload", data))
    if not payload:
        raise APIError(415, "invalid_image", "no valid images in the request", rejected)
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    job_id = svc.jobs.submit_index_job(payload, collection, tag_list)
    if rejected:
        svc.catalog.update_job(job_id, error=json.dumps({"rejected": rejected}))
    return IndexJobAccepted(job_id=job_id, status_url=f"{API}/jobs/{job_id}",
                            accepted_files=len(payload))


@router.get("/jobs/{job_id}", response_model=JobInfo, responses=WRITE_ERRORS, tags=["index"],
            dependencies=[Depends(writable)])
async def get_job(job_id: str, svc: Svc) -> JobInfo:
    job = svc.catalog.get_job(job_id)
    if job is None:
        raise APIError(404, "not_found", f"job '{job_id}' not found")
    return JobInfo(**job)


@router.get("/images/{image_id}", response_model=ImageDetail, responses=ERRORS, tags=["images"])
async def get_image(image_id: int, svc: Svc) -> ImageDetail:
    rec = svc.catalog.get(image_id)
    if rec is None or rec.deleted:
        raise APIError(404, "not_found", f"image {image_id} not found")
    indexed = [m for m, rt in svc.manager.runtimes.items()
               if rt.index is not None and rt.index.has(image_id)]
    return ImageDetail(metadata=ImageMeta(**rec.to_public()),
                       captions=svc.catalog.captions_for(image_id),
                       thumbnail_url=f"{API}/images/{image_id}/thumbnail",
                       image_url=f"{API}/images/{image_id}/file", indexed_in=indexed)


def _image_file(svc: Services, image_id: int, thumb: bool) -> FileResponse:
    rec = svc.catalog.get(image_id)
    if rec is None or rec.deleted:
        raise APIError(404, "not_found", f"image {image_id} not found")
    cache = "public, max-age=31536000, immutable"
    path = svc.settings.images_dir / rec.path
    media = f"image/{(rec.format or 'jpeg').lower()}"
    if rec.thumb_path and (thumb or not path.exists()):
        # Deployments may ship thumbnails only; then the thumbnail stands in for the
        # original, cached briefly in case the original appears later.
        if not thumb:
            cache = "public, max-age=3600"
        path, media = svc.settings.thumbs_dir / rec.thumb_path, "image/jpeg"
    if not path.exists():
        raise APIError(404, "not_found", f"file for image {image_id} is missing")
    # Content-addressed files never change for a given id: cache aggressively.
    return FileResponse(path, media_type=media, headers={
        "Cache-Control": cache, "ETag": f'"{rec.sha256}"'})


@router.get("/images/{image_id}/thumbnail", tags=["images"], responses=ERRORS)
async def get_thumbnail(image_id: int, svc: Svc):
    return _image_file(svc, image_id, thumb=True)


@router.get("/images/{image_id}/file", tags=["images"], responses=ERRORS)
async def get_file(image_id: int, svc: Svc):
    return _image_file(svc, image_id, thumb=False)


@router.delete("/images/{image_id}", responses=WRITE_ERRORS, tags=["images"], status_code=202,
               dependencies=[Depends(writable)])
async def delete_image(image_id: int, svc: Svc) -> dict[str, Any]:
    """Soft-delete (tombstone) an image; indexes drop it in a background sync."""
    if not svc.catalog.mark_deleted(image_id):
        raise APIError(404, "not_found", f"image {image_id} not found")
    job_id = svc.jobs.submit_sync_job(f"delete:{image_id}")
    return {"deleted": image_id, "job_id": job_id, "status_url": f"{API}/jobs/{job_id}"}


@router.get("/health", tags=["ops"])
async def health(svc: Svc) -> dict[str, Any]:
    models = {}
    ok = True
    for key, st in svc.manager.status().items():
        ready = st["index"] is not None and not st["stale"]
        ok &= ready
        models[key] = {"ready": ready, "stale": st["stale"],
                       "indexed": st["index"]["ntotal"] if st["index"] else 0,
                       "modalities": st["model"]["modalities"], "device": st["model"]["device"]}
    return {"status": "ok" if ok else "degraded", "version": svc.version,
            "catalog_images": svc.catalog.count(), "index_type": svc.manager.kind,
            "embed_runtime": svc.settings.embed_runtime, "read_only": svc.settings.read_only,
            "models": models}


@router.get("/metrics", tags=["ops"])
async def metrics(svc: Svc, format: Annotated[str, Query(pattern="^(json|prometheus)$")] = "json"):
    gauges = {"catalog_images": svc.catalog.count()}
    for key, rt in svc.manager.runtimes.items():
        gauges[f"index_vectors{{model=\"{key}\"}}"] = len(rt.index) if rt.index else 0
    if format == "prometheus":
        return PlainTextResponse(svc.metrics.prometheus(gauges))
    snap = svc.metrics.snapshot()
    snap["indexes"] = {k: (v["index"] or {}) | {"disk_bytes": v["manifest"].get("disk_bytes")}
                       for k, v in svc.manager.status().items()}
    snap["catalog_images"] = gauges["catalog_images"]
    return snap


@router.get("/models", tags=["ops"])
async def models(svc: Svc) -> dict[str, Any]:
    st = svc.manager.status()
    return {
        "default_text_model": svc.settings.default_text_model,
        "default_image_model": svc.settings.default_image_model,
        "index_type": svc.manager.kind,
        "embed_runtime": svc.settings.embed_runtime,
        # read_only: no /index/images, DELETE /images/{id} or /jobs (they return 403).
        "read_only": svc.settings.read_only,
        "models": [{**v["model"], "index": v["index"], "stale": v["stale"]} for v in st.values()],
        "calibrations": svc.engine.calibrations.describe(),
        "rerankers": {"text": ["none", "prf", "visual_prf"] +
                      (["cross_encoder"] if svc.engine.cross_encoder else []),
                      "image": ["none", "prf"]},
        "fusions": ["zscore", "minmax", "rrf", "early"],
        "limits": {"max_k": svc.settings.max_k, "max_upload_mb": svc.settings.max_upload_mb,
                   "max_batch_files": svc.settings.max_batch_files},
    }


@router.get("/facets", tags=["images"])
async def facets(svc: Svc) -> dict[str, Any]:
    return svc.catalog.facets()


@router.get("/experiments", tags=["experiments"])
async def experiments(svc: Svc) -> dict[str, Any]:
    """Machine-readable benchmark results (written by ``scripts/run_experiments.py``)."""
    d = svc.settings.results_path
    out = {}
    if d.exists():
        for p in sorted(d.glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
    return {"results": out}


# ---------------------------------------------------------------------- app
def create_app(settings: Settings | None = None, svc: Services | None = None) -> FastAPI:
    settings = settings or (svc.settings if svc else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "services", None) is None:
            app.state.services = await asyncio.to_thread(Services.build, settings)
        yield
        app.state.services.close()

    app = FastAPI(title="VisualSearch API", version="0.1.0", lifespan=lifespan,
                  description="Multimodal semantic image retrieval (CLIP + DINOv2 + FAISS).")
    app.state.services = svc
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                       allow_headers=["*"])

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        t = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        name = getattr(route, "path", "unmatched")
        state = getattr(request.app.state, "services", None)
        if state is not None:
            state.metrics.observe_request(f"{request.method} {name}", response.status_code,
                                          1000 * (time.perf_counter() - t))
        return response

    install_error_handlers(app)
    app.include_router(router)
    return app
