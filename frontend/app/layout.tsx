import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GeoStrom AI — Vertical Slice",
  description:
    "Retrospective tropical cyclone research prototype. Phase 3 vertical slice.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg-base font-sans text-text-primary antialiased">
        {children}
      </body>
    </html>
  );
}
