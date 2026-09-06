import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import SiteFooter from "@/components/layout/SiteFooter";
import SiteHeader from "@/components/layout/SiteHeader";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
// Scientific/technical readouts (coordinates, timestamps, wind speed,
// pressure, model version, storm ID) get a real monospace face rather than
// the system fallback stack `font-mono` used before -- via `next/font`,
// the same zero-extra-dependency mechanism already used for Inter above.
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"], variable: "--font-jetbrains-mono", display: "swap", weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "GeoStrom AI — Retrospective Cyclone Intelligence",
  description:
    "GeoStrom AI studies historical tropical cyclone structure, intensity, and track behaviour " +
    "using satellite observations and machine learning. A retrospective research prototype — " +
    "not an operational forecasting system.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="relative min-h-screen bg-bg-base font-sans text-text-primary antialiased">
        {/* Subtle, fixed atmospheric backdrop -- restrained, per the design
            direction ("subtle gradients", never "excessive neon"). Purely
            decorative; never carries information. A faint drifting grid
            layer (motion-safe only) reinforces the "geospatial/technical
            system" identity without ever being mistaken for real data. */}
        <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 bg-radial-fade" />
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 -z-10 bg-grid-fine bg-grid opacity-[0.35] motion-safe:animate-grid-drift"
        />
        <div className="flex min-h-screen flex-col">
          <SiteHeader />
          <div className="flex-1">{children}</div>
          <SiteFooter />
        </div>
      </body>
    </html>
  );
}
