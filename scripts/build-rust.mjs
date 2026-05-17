import { spawn } from "node:child_process";
import { copyFile } from "node:fs/promises";
import path from "node:path";

const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const useCargoAndroidFallback =
  process.platform === "android" && !process.env.ANDROID_NDK_LATEST_HOME;

const workspaces = [
  {
    name: "@oxide-js/core",
    workspace: "packages/core",
    rustDir: "packages/core/src-rust",
    rustLibrary: "libml_native.so",
    nodeFile: "oxide-native.android-arm64.node"
  },
  {
    name: "@oxide-js/layers",
    workspace: "packages/layers",
    rustDir: "packages/layers/src-rust",
    rustLibrary: "liblayers_native.so",
    nodeFile: "layers-native.android-arm64.node"
  }
];

function runBuild({ name, workspace }) {
  return new Promise((resolve) => {
    const child = spawn(npm, ["run", "build:rust", "-w", workspace], {
      stdio: ["ignore", "pipe", "pipe"]
    });

    const prefix = `[rust:${name}]`;
    child.stdout.on("data", (chunk) => {
      process.stdout.write(
        chunk
          .toString()
          .split(/\r?\n/)
          .filter(Boolean)
          .map((line) => `${prefix} ${line}`)
          .join("\n") + "\n"
      );
    });

    child.stderr.on("data", (chunk) => {
      process.stderr.write(
        chunk
          .toString()
          .split(/\r?\n/)
          .filter(Boolean)
          .map((line) => `${prefix} ${line}`)
          .join("\n") + "\n"
      );
    });

    child.on("close", (code) => {
      resolve({ name, code: code ?? 1 });
    });

    child.on("error", (error) => {
      console.error(`${prefix} ${error.message}`);
      resolve({ name, code: 1 });
    });
  });
}

function runCargoAndroidBuild(workspace) {
  return new Promise((resolve) => {
    const child = spawn("cargo", ["build", "--release"], {
      cwd: workspace.rustDir,
      stdio: ["ignore", "pipe", "pipe"]
    });

    const prefix = `[rust:${workspace.name}]`;
    child.stdout.on("data", (chunk) => {
      process.stdout.write(
        chunk
          .toString()
          .split(/\r?\n/)
          .filter(Boolean)
          .map((line) => `${prefix} ${line}`)
          .join("\n") + "\n"
      );
    });

    child.stderr.on("data", (chunk) => {
      process.stderr.write(
        chunk
          .toString()
          .split(/\r?\n/)
          .filter(Boolean)
          .map((line) => `${prefix} ${line}`)
          .join("\n") + "\n"
      );
    });

    child.on("close", (code) => {
      resolve({ name: workspace.name, code: code ?? 1, workspace });
    });

    child.on("error", (error) => {
      console.error(`${prefix} ${error.message}`);
      resolve({ name: workspace.name, code: 1, workspace });
    });
  });
}

async function copyAndroidNodeFile(result) {
  if (result.code !== 0) return result;

  const source = path.join(
    result.workspace.rustDir,
    "target",
    "release",
    result.workspace.rustLibrary
  );
  const target = path.join(result.workspace.workspace, result.workspace.nodeFile);

  await copyFile(source, target);
  console.log(`[rust:${result.name}] wrote ${target}`);

  return result;
}

if (useCargoAndroidFallback && process.arch !== "arm64") {
  console.error(`[rust] Android cargo fallback only supports arm64. Current arch: ${process.arch}`);
  process.exit(1);
}

if (useCargoAndroidFallback) {
  console.warn("[rust] ANDROID_NDK_LATEST_HOME is missing; using local cargo Android fallback.");
}

const results = useCargoAndroidFallback
  ? await Promise.all(workspaces.map(runCargoAndroidBuild)).then((items) => Promise.all(items.map(copyAndroidNodeFile)))
  : await Promise.all(workspaces.map(runBuild));
const failed = results.filter((result) => result.code !== 0);

if (failed.length > 0) {
  for (const result of failed) {
    console.error(`[rust:${result.name}] failed with exit code ${result.code}`);
  }
  process.exit(1);
}
