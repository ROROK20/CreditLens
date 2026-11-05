/**
 * Which engine URL the page calls, or null for none (demo mode).
 *
 * `NEXT_PUBLIC_API_URL` is inlined by Next at BUILD time, so setting it on a
 * running server does nothing. When it is set, that URL is used. When it is
 * unset, only `next dev` (NODE_ENV "development") falls back to
 * http://localhost:8002, the local engine; a production build gets null, makes
 * no engine call at all and runs the engine's recorded model. Falling back to
 * localhost in a production build would make a public deploy without the variable call
 * each VISITOR's own localhost:8002.
 *
 * @param {string | undefined} configured NEXT_PUBLIC_API_URL
 * @param {string | undefined} nodeEnv NODE_ENV
 * @returns {string | null}
 */
export function resolveApiUrl(configured, nodeEnv) {
  const url = (configured ?? "").trim();
  if (url) return url.replace(/\/$/, "");
  return nodeEnv === "development" ? "http://localhost:8002" : null;
}
