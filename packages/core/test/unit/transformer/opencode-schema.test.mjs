import assert from "node:assert/strict";
import test from "node:test";
import { OpencodeGoTransformer } from "@ccr/core/transformer/opencodego.transformer.ts";

test("cleanJsonSchema strips ref pointers but preserves arrays (deepseek prefixItems regression)", () => {
  const t = new OpencodeGoTransformer();
  const out = t.cleanJsonSchema({
    type: "object",
    $ref: "#/definitions/X",
    $schema: "http://json-schema.org/draft-07/schema#",
    definitions: { X: {} },
    properties: {
      items: {
        type: "array",
        prefixItems: [{ type: "string" }, { type: "number" }],
        items: { type: "string" },
        enum: ["a", "b"],
      },
    },
    required: ["items"],
  });

  assert.equal(out.$ref, undefined);
  assert.equal(out.$schema, undefined);
  assert.equal(out.definitions, undefined);
  assert.ok(Array.isArray(out.properties.items.prefixItems), "prefixItems stays an array");
  assert.deepEqual(out.properties.items.prefixItems, [{ type: "string" }, { type: "number" }]);
  assert.deepEqual(out.properties.items.enum, ["a", "b"]);
  assert.deepEqual(out.required, ["items"]);
});

test("cleanJsonSchema normalizes required/enum unions without mangling", () => {
  const t = new OpencodeGoTransformer();
  const out = t.cleanJsonSchema({
    type: "object",
    properties: { x: { type: "string" } },
    required: "x",
    anyOf: [{ type: "string" }, { type: "number" }],
  });
  assert.deepEqual(out.required, ["x"]);
  assert.equal(out.anyOf.length, 2);
});
