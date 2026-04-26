function toGoogleType(type: string): string {
  if (!type || typeof type !== "string") return type;
  const typeMap: Record<string, string> = {
    string: "STRING",
    number: "NUMBER",
    integer: "INTEGER",
    boolean: "BOOLEAN",
    array: "ARRAY",
    object: "OBJECT",
    null: "STRING",
  };
  return typeMap[type.toLowerCase()] || type.toUpperCase();
}

export function sanitizeSchema(schema: any): any {
  if (!schema || typeof schema !== "object") {
    return { type: "OBJECT", properties: {}, required: [] };
  }

  const cleaned: any = { type: "OBJECT", properties: {}, required: [] };

  if (schema.type && schema.type !== "object") {
    return schema;
  }

  if (schema.properties && typeof schema.properties === "object") {
    cleaned.properties = {};
    for (const [key, prop] of Object.entries(schema.properties)) {
      cleaned.properties[key] = sanitizeProperty(prop);
    }
  }

  if (
    Array.isArray(schema.required) &&
    schema.required.every((r: any) => typeof r === "string")
  ) {
    cleaned.required = schema.required;
  }

  return cleaned;
}

function sanitizeProperty(prop: any): any {
  if (!prop || typeof prop !== "object") {
    return { type: "STRING" };
  }

  const cleaned: any = {};

  if (prop.type) {
    cleaned.type = toGoogleType(prop.type);
  } else if (prop.properties || prop.items) {
    cleaned.type = "OBJECT";
  }

  if (prop.description) cleaned.description = prop.description;
  if (prop.enum) cleaned.enum = prop.enum;
  if (prop.default !== undefined) cleaned.default = prop.default;
  if (prop.nullable) cleaned.nullable = prop.nullable;

  const typeLower = (prop.type || "").toLowerCase();

  if ((typeLower === "object" || cleaned.type === "OBJECT") && prop.properties) {
    cleaned.properties = {};
    for (const [key, val] of Object.entries(prop.properties)) {
      cleaned.properties[key] = sanitizeProperty(val);
    }
  }

  if ((typeLower === "array" || cleaned.type === "ARRAY")) {
    if (prop.items) {
      cleaned.items = sanitizeProperty(prop.items);
    } else {
      cleaned.items = { type: "STRING" };
    }
  }

  if (prop.$ref) {
    return { type: "OBJECT", description: prop.description || "" };
  }

  delete cleaned.$schema;
  delete cleaned.defs;
  delete cleaned.definitions;

  return Object.keys(cleaned).length > 0 ? cleaned : { type: "STRING" };
}

export function cleanSchema(schema: any): any {
  if (!schema) return { type: "OBJECT", properties: {}, required: [] };

  const result: any = { type: "OBJECT", properties: {}, required: [] };

  if (schema.properties && typeof schema.properties === "object") {
    for (const [key, prop] of Object.entries(schema.properties)) {
      result.properties[key] = cleanProperty(prop);
    }
  }

  if (Array.isArray(schema.required)) {
    result.required = schema.required.filter((r: any) => typeof r === "string");
  }

  return result;
}

function cleanProperty(prop: any): any {
  if (!prop || typeof prop !== "object") return { type: "STRING" };

  const result: any = {};

  if (prop.type) {
    result.type = toGoogleType(prop.type);
  }

  if (prop.description) {
    result.description = prop.description;
  }

  if (prop.enum) {
    result.enum = prop.enum;
  }

  if (prop.nullable) {
    result.nullable = prop.nullable;
  }

  const typeLower = (prop.type || "").toLowerCase();

  if ((typeLower === "object" || result.type === "OBJECT") && prop.properties) {
    result.properties = {};
    for (const [key, val] of Object.entries(prop.properties)) {
      result.properties[key] = cleanProperty(val);
    }
  }

  if ((typeLower === "array" || result.type === "ARRAY")) {
    if (prop.items) {
      result.items = cleanProperty(prop.items);
    } else {
      result.items = { type: "STRING" };
    }
  }

  delete result.$ref;
  delete result.$schema;
  delete result.anyOf;
  delete result.oneOf;
  delete result.allOf;
  delete result.defs;
  delete result.definitions;
  delete result.additionalProperties;

  return result;
}

export function cleanCacheControl(messages: any[]): any[] {
  return messages.map((msg) => {
    const cleaned = { ...msg };

    if (cleaned.cache_control) {
      delete cleaned.cache_control;
    }

    if (Array.isArray(cleaned.content)) {
      cleaned.content = cleaned.content.map((block: any) => {
        if (block?.cache_control) {
          const { cache_control: _, ...rest } = block;
          return rest;
        }
        return block;
      });
    }

    return cleaned;
  });
}