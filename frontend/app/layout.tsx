import type { Metadata } from "next";
import "./globals.css";
import { resolveSiteUrl } from "@/lib/siteUrl.mjs";

// Absolute og / twitter URLs need the site's public origin. Without
// NEXT_PUBLIC_SITE_URL a production build ships no preview image and no og:url
// rather than ones pointing at localhost (lib/siteUrl.mjs).
const SITE_URL = resolveSiteUrl(process.env.NEXT_PUBLIC_SITE_URL, process.env.NODE_ENV);
const OG_IMAGES = SITE_URL
  ? [{ url: "/og.png", width: 1200, height: 630, alt: "CreditLens explainable credit scoring" }]
  : [];

// True of both builds: with an engine the page shows its Shapley values and live audit,
// without one its recorded model, log-odds factors and recorded audit.
const description =
  "Explainable credit scoring on synthetic data: a FICO score plus cash-flow signals, the factors behind each decision, and a fairness audit that states its uncertainty.";

export const metadata: Metadata = {
  ...(SITE_URL ? { metadataBase: new URL(SITE_URL) } : {}),
  title: {
    default: "CreditLens — Explainable Credit Scoring",
    template: "%s · CreditLens",
  },
  description,
  applicationName: "CreditLens",
  authors: [{ name: "Rohan Khiani", url: "https://github.com/ROROK20" }],
  keywords: [
    "credit scoring",
    "explainable AI",
    "SHAP",
    "fairness audit",
    "alternative data",
    "FastAPI",
  ],
  icons: {
    icon: "/icon.svg",
  },
  openGraph: {
    type: "website",
    siteName: "CreditLens",
    title: "CreditLens — Explainable Credit Scoring",
    description,
    ...(SITE_URL ? { url: "/" } : {}),
    images: OG_IMAGES,
  },
  twitter: {
    card: SITE_URL ? "summary_large_image" : "summary",
    title: "CreditLens — Explainable Credit Scoring",
    description,
    images: OG_IMAGES.map((image) => image.url),
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-mono antialiased">{children}</body>
    </html>
  );
}
