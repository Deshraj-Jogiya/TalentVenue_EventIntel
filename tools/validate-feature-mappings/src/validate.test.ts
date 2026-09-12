import { test } from "node:test";
import * as assert from "node:assert";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { validateFeatureMappingsFile } from "./validate";

function writeTempJson(content: unknown): string {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "fm-test-")), "feature_mappings.json");
  fs.writeFileSync(file, JSON.stringify(content));
  return file;
}

test("accepts a well-formed feature_mappings.json", () => {
  const file = writeTempJson({
    global_cancellation_rate: 0.1758,
    presenters: { "1000002": 0.22 },
    artists: { "501": 0.3 },
  });
  const result = validateFeatureMappingsFile(file);
  assert.strictEqual(result.ok, true);
  if (result.ok) {
    assert.strictEqual(result.presenterCount, 1);
    assert.strictEqual(result.artistCount, 1);
  }
});

test("rejects a rate outside [0, 1]", () => {
  const file = writeTempJson({
    global_cancellation_rate: 1.5,
    presenters: {},
    artists: {},
  });
  const result = validateFeatureMappingsFile(file);
  assert.strictEqual(result.ok, false);
});

test("rejects a missing required field", () => {
  const file = writeTempJson({ global_cancellation_rate: 0.2, presenters: {} });
  const result = validateFeatureMappingsFile(file);
  assert.strictEqual(result.ok, false);
});

test("rejects malformed JSON", () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "fm-test-")), "bad.json");
  fs.writeFileSync(file, "{not valid json");
  const result = validateFeatureMappingsFile(file);
  assert.strictEqual(result.ok, false);
});
