import { useId, useState, type DragEvent } from "react";
import { api, ApiError, type JobInfo } from "../api";
import { useApp } from "../AppContext";
import { ErrorView } from "../components/ErrorView";
import { checkImageFile } from "../components/ImageInput";
import { parseCsv } from "../lib/filters";
import { fmtBytes, fmtInt } from "../lib/format";
import { useInterval } from "../lib/hooks";
import { navigate } from "../lib/route";

const COLLECTION_RE = /^[\w-]+$/;

const COUNT_LABELS: [string, string][] = [
  ["added", "Added"],
  ["duplicate", "Duplicate"],
  ["near_duplicate_skipped", "Near-duplicate"],
  ["invalid", "Invalid"],
];

function parseRejected(err: string | null | undefined): { file: string; error: string }[] {
  if (!err) return [];
  try {
    const v = JSON.parse(err) as { rejected?: { file: string; error: string }[] };
    return Array.isArray(v.rejected) ? v.rejected : [];
  } catch {
    return [];
  }
}

export function IndexView({ active }: { active: boolean }) {
  const { models, refreshFacets, requestSearch, reload } = useApp();
  const id = useId();
  const [files, setFiles] = useState<File[]>([]);
  const [rejected, setRejected] = useState<string[]>([]);
  const [collection, setCollection] = useState("uploads");
  const [tags, setTags] = useState("");
  const [over, setOver] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [accepted, setAccepted] = useState(0);
  const [job, setJob] = useState<JobInfo | null>(null);
  const [jobCollection, setJobCollection] = useState("");

  const maxMb = models?.limits.max_upload_mb ?? 10;
  const maxFiles = models?.limits.max_batch_files ?? 256;
  const collectionOk = COLLECTION_RE.test(collection) && collection.length <= 64;
  const done = job?.status === "succeeded" || job?.status === "failed";

  const addFiles = (list: FileList | File[] | null) => {
    if (!list) return;
    const ok: File[] = [];
    const bad: string[] = [];
    for (const f of Array.from(list)) {
      const p = checkImageFile(f, maxMb);
      if (p) bad.push(p);
      else ok.push(f);
    }
    setFiles((cur) => {
      const key = (f: File) => `${f.name}:${f.size}:${f.lastModified}`;
      const seen = new Set(cur.map(key));
      return [...cur, ...ok.filter((f) => !seen.has(key(f)))];
    });
    setRejected(bad);
  };

  useInterval(async (signal) => {
    if (!jobId) return;
    try {
      const j = await api.job(jobId, signal);
      setJob(j);
      if (j.status === "succeeded" || j.status === "failed") {
        setJobId(null);
        refreshFacets();
        reload();
      }
    } catch (e) {
      if (!signal.aborted) setError(e);
    }
  }, 1000, !!jobId && active);

  const submit = async () => {
    if (!files.length || !collectionOk) return;
    if (files.length > maxFiles) {
      setError(new ApiError(413, "too_many_files", `At most ${maxFiles} files per request.`));
      return;
    }
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    form.append("collection", collection);
    const t = parseCsv(tags);
    if (t.length) form.append("tags", t.join(","));
    setSubmitting(true);
    setError(null);
    setJob(null);
    try {
      const r = await api.indexImages(form);
      setJobId(r.job_id);
      setAccepted(r.accepted_files);
      setJobCollection(collection);
      setFiles([]);
    } catch (e) {
      setError(e);
    } finally {
      setSubmitting(false);
    }
  };

  const totalBytes = files.reduce((a, f) => a + f.size, 0);
  const counts = job?.result?.counts ?? {};
  const serverRejected = parseRejected(job?.error);
  const pct = job && job.total ? Math.round((100 * job.progress) / job.total) : 0;

  if (models?.read_only) {
    return (
      <div className="view narrow">
        <section className="panel">
          <h2>Add images to the index</h2>
          <p className="hint">This server is read-only: indexing new images is turned off.</p>
        </section>
      </div>
    );
  }

  return (
    <div className="view narrow">
      <section className="panel">
        <h2>Add images to the index</h2>
        <p className="hint">
          Uploads are validated, de-duplicated (exact and near-duplicate), embedded with every loaded
          model and added to the indexes in a background job.
        </p>
        <label htmlFor={`${id}-files`} className={`dropzone dropzone-lg${over ? " over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e: DragEvent) => { e.preventDefault(); setOver(false); addFiles(e.dataTransfer.files); }}>
          <svg width="32" height="32" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6">
            <path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" />
          </svg>
          <span><strong>Drop images here</strong> or click to choose files</span>
          <span className="hint">Up to {maxFiles} files, {maxMb} MB each</span>
        </label>
        <input id={`${id}-files`} type="file" accept="image/*" multiple className="visually-hidden"
          onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
        {rejected.length > 0 && (
          <ul className="field-error" role="alert">{rejected.map((r, i) => <li key={i}>{r}</li>)}</ul>
        )}
        {files.length > 0 && (
          <div className="file-list">
            <div className="file-list-head">
              <span><strong>{files.length}</strong> file{files.length === 1 ? "" : "s"} · {fmtBytes(totalBytes)}</span>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => setFiles([])}>Clear</button>
            </div>
            <ul>
              {files.slice(0, 200).map((f, i) => (
                <li key={`${f.name}-${i}`}>
                  <span className="file-name">{f.name}</span>
                  <span className="muted mono">{fmtBytes(f.size)}</span>
                  <button type="button" className="btn btn-ghost btn-icon btn-sm" aria-label={`Remove ${f.name}`}
                    onClick={() => setFiles((cur) => cur.filter((_, j) => j !== i))}>×</button>
                </li>
              ))}
            </ul>
            {files.length > maxFiles && <p className="field-error">Too many files: at most {maxFiles} per upload.</p>}
          </div>
        )}
        <div className="config-grid">
          <div className="field">
            <label htmlFor={`${id}-col`}>Collection</label>
            <input id={`${id}-col`} value={collection} maxLength={64} aria-invalid={!collectionOk}
              aria-describedby={`${id}-col-help`} onChange={(e) => setCollection(e.target.value)} />
            <span id={`${id}-col-help`} className={collectionOk ? "hint" : "field-error"}>
              Letters, digits, “_” and “-” only.
            </span>
          </div>
          <div className="field field-wide">
            <label htmlFor={`${id}-tags`}>Tags</label>
            <input id={`${id}-tags`} value={tags} placeholder="comma-separated, e.g. holiday, 2024"
              onChange={(e) => setTags(e.target.value)} />
          </div>
        </div>
        <button className="btn btn-primary" disabled={!files.length || !collectionOk || submitting || !!jobId || files.length > maxFiles}
          onClick={submit}>
          {submitting ? "Uploading…" : `Index ${files.length || ""} image${files.length === 1 ? "" : "s"}`}
        </button>
      </section>

      {error != null && <ErrorView error={error} />}

      {(jobId || job) && (
        <section className="panel" aria-live="polite">
          <h2>Indexing job</h2>
          <p className="hint mono">{job?.id ?? jobId} · {accepted} file{accepted === 1 ? "" : "s"} accepted</p>
          <div className="progress-row">
            <span className={`status-pill status-${job?.status ?? "queued"}`}>{job?.status ?? "queued"}</span>
            <progress max={job?.total || 1} value={job?.progress ?? 0} aria-label="Indexing progress" />
            <span className="mono">{fmtInt(job?.progress ?? 0)} / {fmtInt(job?.total ?? accepted)} ({pct}%)</span>
          </div>
          {done && job?.status === "succeeded" && (
            <>
              <dl className="counts">
                {COUNT_LABELS.map(([k, label]) => (
                  <div key={k} className={`count count-${k}`}><dt>{label}</dt><dd>{fmtInt(counts[k] ?? 0)}</dd></div>
                ))}
              </dl>
              <button className="btn btn-primary" onClick={() => {
                requestSearch({ filters: { collections: [jobCollection] }, query: { mode: "text" }, run: false });
                navigate("search");
              }}>
                Search this collection
              </button>
            </>
          )}
          {job?.status === "failed" && job.error && !serverRejected.length && (
            <div className="notice notice-error"><strong>Job failed</strong><p>{job.error}</p></div>
          )}
          {serverRejected.length > 0 && (
            <details open>
              <summary>{serverRejected.length} file{serverRejected.length === 1 ? "" : "s"} rejected by the server</summary>
              <ul className="issues">{serverRejected.map((r, i) => <li key={i}><code>{r.file}</code>: {r.error}</li>)}</ul>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
