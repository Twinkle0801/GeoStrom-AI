import { FlatCompat } from "@eslint/eslintrc";

const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    // lib/api-types.ts: generated file, never hand-edited.
    // .next/, node_modules/: build/dependency output -- `next lint` ignores
    // these by default, but a raw `eslint .` invocation (this project's
    // `npm run lint` script) does not inherit that default automatically
    // under flat config, so it must be listed explicitly. Found for real:
    // a `.next/` produced by `npm run dev`/`build` made `npm run lint`
    // report 1300+ spurious errors against compiled webpack output.
    ignores: ["lib/api-types.ts", ".next/**", "node_modules/**"],
  },
];

export default eslintConfig;
