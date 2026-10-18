export function JsonView({ value, summary = "Raw JSON" }: { value: unknown; summary?: string }) {
  return (
    <details className="json-view">
      <summary>{summary}</summary>
      <pre className="json">{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}
