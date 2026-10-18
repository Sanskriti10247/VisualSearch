import { ApiError, validationIssues } from "../api";

interface Props {
  error: unknown;
  onRetry?: () => void;
  compact?: boolean;
}

/** Renders an API error envelope (code + message + validation details) or an unreachable notice. */
export function ErrorView({ error, onRetry, compact }: Props) {
  const e = error instanceof ApiError ? error : new ApiError(0, "unknown", String(error));
  if (e.unreachable) {
    return (
      <div className={`notice notice-error${compact ? " compact" : ""}`} role="alert">
        <strong>Backend unreachable</strong>
        <p>
          {e.message} Start it with <code>visualsearch serve</code> (default
          {" "}<code>http://127.0.0.1:8000</code>); the dev server proxies <code>/api</code> there.
        </p>
        {onRetry && <button className="btn" onClick={onRetry}>Retry</button>}
      </div>
    );
  }
  const issues = validationIssues(e.details);
  const otherDetails = !issues.length && e.details != null ? e.details : null;
  return (
    <div className={`notice notice-error${compact ? " compact" : ""}`} role="alert">
      <strong>
        <code className="err-code">{e.code}</code>
        {e.status ? <span className="muted"> · HTTP {e.status}</span> : null}
      </strong>
      <p>{e.message}</p>
      {issues.length > 0 && (
        <ul className="issues">
          {issues.map((i, n) => (
            <li key={n}>
              <code>{i.loc.filter((l) => l !== "body").join(".") || "request"}</code>: {i.msg}
            </li>
          ))}
        </ul>
      )}
      {otherDetails !== null && (
        <details>
          <summary>Details</summary>
          <pre className="json">{typeof otherDetails === "string" ? otherDetails : JSON.stringify(otherDetails, null, 2)}</pre>
        </details>
      )}
      {onRetry && <button className="btn" onClick={onRetry}>Try again</button>}
    </div>
  );
}
