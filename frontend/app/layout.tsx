import type { Metadata } from "next";
import { Inter } from "next/font/google";
import SiteFooter from "@/components/layout/SiteFooter";
import SiteHeader from "@/components/layout/SiteHeader";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });

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
    <html lang="en" className={inter.variable}>
      <body className="relative min-h-screen bg-bg-base font-sans text-text-primary antialiased">
        {/* Subtle, fixed atmospheric backdrop -- restrained, per the design
            direction ("subtle gradients", never "excessive neon"). Purely
            decorative; never carries information. */}
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(80%_60%_at_50%_-10%,rgba(76,141,255,0.10),transparent_60%)]"
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
