export function JsonView({ value, summary = "Raw JSON", open }: { value: unknown; summary?: string; open?: boolean }) {
  return (
    <details className="json-view" open={open}>
      <summary>{summary}</summary>
      <pre className="json">{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}
