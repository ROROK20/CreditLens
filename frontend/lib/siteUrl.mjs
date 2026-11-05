/**
 * The site's public origin for absolute social-preview URLs, or null.
 *
 * `NEXT_PUBLIC_SITE_URL` (inlined at build) when set; otherwise
 * http://localhost:3003 under `next dev` only. A production build without it
 * gets null: layout.tsx then sets no metadataBase, no og:url and no og / twitter
 * image, since Next resolves relative ones against localhost when the base is
 * missing, which would advertise http://localhost:3003/og.png to every crawler.
 *
 * @param {string | undefined} configured NEXT_PUBLIC_SITE_URL
 * @param {string | undefined} nodeEnv NODE_ENV
 * @returns {string | null}
 */
export function resolveSiteUrl(configured, nodeEnv) {
  const url = (configured ?? "").trim();
  if (url) return url.replace(/\/$/, "");
  return nodeEnv === "development" ? "http://localhost:3003" : null;
}
