import type { JSONSchema7 } from "json-schema";

export function providerizeSchema(
  provider: string,
  schema: JSONSchema7
): JSONSchema7 {
  // Handle primitive types or schemas without properties
  if (
    !schema ||
    typeof schema !== "object" ||
    schema.type !== "object" ||
    !schema.properties
  ) {
    return schema;
  }

  const processedProperties: Record<string, JSONSchema7> = {};

  // Recursively process each property
  for (const [key, property] of Object.entries(schema.properties)) {
    if (typeof property === "object" && property !== null) {
      let processedProperty = property as JSONSchema7;

      // Remove uri format for OpenAI and Google
      if ((provider === "openai" || provider === "google") && processedProperty.format === "uri") {
        processedProperty = { ...processedProperty };
        delete processedProperty.format;
      }

      if (processedProperty.type === "object") {
        // Recursively process nested objects
        processedProperties[key] = providerizeSchema(
          provider,
          processedProperty
        );
      } else if (
        processedProperty.type === "array" &&
        processedProperty.items
      ) {
        // Handle arrays with object items
        const items = processedProperty.items;
        if (
          typeof items === "object" &&
          !Array.isArray(items) &&
          items.type === "object"
        ) {
          processedProperties[key] = {
            ...processedProperty,
            items: providerizeSchema(provider, items as JSONSchema7),
          };
        } else {
          processedProperties[key] = processedProperty;
        }
      } else {
        processedProperties[key] = processedProperty;
      }
    } else {
      // Handle boolean properties (true/false schemas)
      processedProperties[key] = property as unknown as JSONSchema7;
    }
  }

  const result: JSONSchema7 = {
    ...schema,
    properties: processedProperties,
  };

  // Only add required properties for OpenAI
  if (provider === "openai") {
    result.required = Object.keys(schema.properties);
    result.additionalProperties = false;
  }

  return result;
}
