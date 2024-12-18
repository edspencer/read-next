import typescript from "rollup-plugin-typescript2";
// import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import peerDepsExternal from "rollup-plugin-peer-deps-external";

export default [
  {
    input: "src/index.ts",
    output: {
      file: "dist/index.js",
      format: "esm",
      sourcemap: true,
    },
    plugins: [peerDepsExternal(), typescript({ useTsconfigDeclarationDir: true }), commonjs()],
    watch: {
      include: "src/**",
    },
  },
  // CommonJS Build
  {
    input: "src/index.ts",
    output: {
      file: "dist/index.cjs.js",
      format: "cjs",
      sourcemap: true,
    },
    plugins: [peerDepsExternal(), typescript({ useTsconfigDeclarationDir: true }), commonjs()],
    watch: {
      include: "src/**",
    },
  },
  // {
  //   input: "./src/index.ts",
  //   output: [
  //     {
  //       file: "./dist/index.esm.js",
  //       format: "esm",
  //     },
  //     {
  //       file: "./dist/index.cjs.js",
  //       format: "cjs",
  //     },
  //   ],
  //   plugins: [
  //     peerDepsExternal(), // Automatically externalize peer dependencies
  //     // resolve(), // Resolve modules from node_modules
  //     commonjs(), // Convert CommonJS modules to ES6
  //     typescript({
  //       tsconfig: "./tsconfig.json",
  //       exclude: ["src/test/**/*", "src/script/**/*"], // Exclude test files from being bundled
  //     }),
  //   ],
  //   watch: {
  //     include: "src/**",
  //   },
  // },
];
