/** "About these scores" explainer (collapsible). */
export function ScoreInfo() {
  return (
    <details className="score-info">
      <summary>
        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.5v.5" strokeLinecap="round" />
        </svg>
        About these scores
      </summary>
      <ul>
        <li>
          <strong>Cosine</strong> is the raw similarity in one model's embedding space. It is only
          comparable within one model and query type: CLIP text→image scores typically sit around
          0.20–0.35, while image→image scores are much higher.
        </li>
        <li>
          <strong>top x%</strong> compares the score against random (query, image) pairs measured on
          validation data: “top 1%” means only 1% of random pairs score this high.
        </li>
        <li>
          <strong>P(match)</strong> is calibrated to Flickr30k annotations, where each caption has exactly
          one annotated image. It therefore underestimates relevance for generic queries that many
          images satisfy.
        </li>
        <li><strong>DINOv2</strong> is image-only (no text encoder): use it for image→image similarity.</li>
        <li>
          <strong>CLIP</strong> is weak at counting, negation (“without”), spatial relations
          (“left of”) and reading text in images.
        </li>
      </ul>
    </details>
  );
}
