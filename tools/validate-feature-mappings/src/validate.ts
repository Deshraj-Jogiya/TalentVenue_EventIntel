import * as fs from "fs";
import * as path from "path";
import { FeatureMappingsSchema } from "./schema";

export function validateFeatureMappingsFile(filePath: string): { ok: true; presenterCount: number; artistCount: number } | { ok: false; error: string } {
  let raw: string;
  try {
    raw = fs.readFileSync(filePath, "utf-8");
  } catch (err) {
    return { ok: false, error: `Could not read ${filePath}: ${(err as Error).message}` };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    return { ok: false, error: `Invalid JSON in ${filePath}: ${(err as Error).message}` };
  }

  const result = FeatureMappingsSchema.safeParse(parsed);
  if (!result.success) {
    return { ok: false, error: result.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ") };
  }

  return {
    ok: true,
    presenterCount: Object.keys(result.data.presenters).length,
    artistCount: Object.keys(result.data.artists).length,
  };
}

function main() {
  const target = process.argv[2] ?? path.join(__dirname, "..", "..", "..", "data", "models", "feature_mappings.json");
  const result = validateFeatureMappingsFile(target);

  if (result.ok) {
    console.log(`[+] ${target} is valid.`);
    console.log(`    presenters: ${result.presenterCount}, artists: ${result.artistCount}`);
    process.exit(0);
  } else {
    console.error(`[X] ${target} failed validation: ${result.error}`);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}
