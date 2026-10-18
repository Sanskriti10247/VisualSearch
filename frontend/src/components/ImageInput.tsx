import { useEffect, useId, useRef, useState, type DragEvent } from "react";
import type { QueryImage } from "../lib/search";
import { fmtBytes } from "../lib/format";

interface Props {
  value: QueryImage | null;
  onChange: (v: QueryImage | null) => void;
  maxMb: number;
  /** Listen for clipboard pastes (only the visible input should). */
  pasteActive: boolean;
  compact?: boolean;
}

export function checkImageFile(file: File, maxMb: number): string | null {
  if (!file.type.startsWith("image/")) return `“${file.name}” is not an image (${file.type || "unknown type"}).`;
  if (file.size > maxMb * 1024 * 1024) {
    return `“${file.name}” is ${fmtBytes(file.size)}; the limit is ${maxMb} MB.`;
  }
  if (file.size === 0) return `“${file.name}” is empty.`;
  return null;
}

/** Query-image picker: drag & drop, click to browse, or paste from clipboard. */
export function ImageInput({ value, onChange, maxMb, pasteActive, compact }: Props) {
  const inputId = useId();
  const fileRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);

  // Revoke object URLs of replaced uploads.
  useEffect(() => () => {
    if (value?.kind === "file") URL.revokeObjectURL(value.previewUrl);
  }, [value]);

  const accept = (file: File | null | undefined) => {
    if (!file) return;
    const problem = checkImageFile(file, maxMb);
    setError(problem);
    if (problem) return;
    onChange({ kind: "file", file, previewUrl: URL.createObjectURL(file) });
  };

  const latestAccept = useRef(accept);
  latestAccept.current = accept;
  useEffect(() => {
    if (!pasteActive) return;
    const onPaste = (e: ClipboardEvent) => {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.kind === "file" && i.type.startsWith("image/"));
      const file = item?.getAsFile();
      if (!file) return;
      e.preventDefault();
      const named = file.name && file.name !== "image.png" ? file : new File([file], `pasted.${file.type.split("/")[1] || "png"}`, { type: file.type });
      latestAccept.current(named);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [pasteActive]);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    accept(e.dataTransfer.files?.[0]);
  };

  if (value) {
    return (
      <div className={`image-preview${compact ? " compact" : ""}`}>
        <img src={value.previewUrl} alt="Query image" />
        <div className="image-preview-meta">
          <span>
            {value.kind === "file"
              ? <>{value.file.name} <span className="muted">· {fmtBytes(value.file.size)}</span></>
              : <>Indexed image <strong>#{value.imageId}</strong></>}
          </span>
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setError(null); onChange(null); }}>
            Remove
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        className={`dropzone${over ? " over" : ""}${compact ? " compact" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <svg width="28" height="28" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6">
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <circle cx="9" cy="10" r="2" />
          <path d="M21 16l-5-5-8 9" />
        </svg>
        <span><strong>Drop an image</strong>, click to browse, or paste (Ctrl/⌘ V)</span>
        <span className="hint">Any image type up to {maxMb} MB</span>
      </label>
      <input
        ref={fileRef}
        id={inputId}
        type="file"
        accept="image/*"
        className="visually-hidden"
        onChange={(e) => { accept(e.target.files?.[0]); e.target.value = ""; }}
      />
      {error && <p className="field-error" role="alert">{error}</p>}
    </div>
  );
}
