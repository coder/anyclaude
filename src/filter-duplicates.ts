import type { AnthropicMessage } from "./anthropic-api-types";
import { isDebugEnabled, isVerboseDebugEnabled } from "./debug";

/**
 * Filters duplicate tool calls and their corresponding tool results from messages.
 * Ensures consistency between tool calls and results while preserving order.
 * 
 * Important: This function maintains the original order of tool calls and results,
 * only removing duplicates while ensuring proper pairing between calls and results.
 */
export function filterDuplicateToolCalls(messages: AnthropicMessage[]): AnthropicMessage[] {
  // Step 1: Identify all tool call IDs and track their first occurrence
  const toolCallFirstOccurrence = new Map<string, { msgIndex: number; contentIndex: number }>();
  const duplicateOccurrences = new Set<string>(); // Format: "msgIndex:contentIndex"
  
  messages.forEach((msg, msgIndex) => {
    if (msg.role === 'assistant' && Array.isArray(msg.content)) {
      msg.content.forEach((item, contentIndex) => {
        if (typeof item === 'object' && item.type === 'tool_use') {
          const first = toolCallFirstOccurrence.get(item.id);
          if (first) {
            // This is a duplicate
            duplicateOccurrences.add(`${msgIndex}:${contentIndex}`);
            if (isVerboseDebugEnabled()) {
              console.error(`[ANYCLAUDE DEBUG] Found duplicate tool_use '${item.id}' at message ${msgIndex}, content ${contentIndex} (first was at ${first.msgIndex}:${first.contentIndex})`);
            }
          } else {
            // First occurrence
            toolCallFirstOccurrence.set(item.id, { msgIndex, contentIndex });
          }
        }
      });
    }
  });

  // Step 2: Track tool results and identify duplicates
  const toolResultFirstOccurrence = new Map<string, { msgIndex: number; contentIndex: number }>();
  const duplicateResults = new Set<string>(); // Format: "msgIndex:contentIndex"
  
  messages.forEach((msg, msgIndex) => {
    if (msg.role === 'user' && Array.isArray(msg.content)) {
      msg.content.forEach((item, contentIndex) => {
        if (typeof item === 'object' && item.type === 'tool_result') {
          const first = toolResultFirstOccurrence.get(item.tool_use_id);
          if (first) {
            // This is a duplicate result
            duplicateResults.add(`${msgIndex}:${contentIndex}`);
            if (isVerboseDebugEnabled()) {
              console.error(`[ANYCLAUDE DEBUG] Found duplicate tool_result for '${item.tool_use_id}' at message ${msgIndex}, content ${contentIndex} (first was at ${first.msgIndex}:${first.contentIndex})`);
            }
          } else {
            // First occurrence
            toolResultFirstOccurrence.set(item.tool_use_id, { msgIndex, contentIndex });
          }
        }
      });
    }
  });
  
  if (isVerboseDebugEnabled() && (duplicateOccurrences.size > 0 || duplicateResults.size > 0)) {
    console.error(`[ANYCLAUDE DEBUG] Summary: Found ${duplicateOccurrences.size} duplicate tool calls and ${duplicateResults.size} duplicate tool results`);
  }

  // Step 3: Filter messages
  const filteredMessages = messages.map((msg, msgIndex) => {
    if (msg.role === 'assistant' && Array.isArray(msg.content)) {
      // Filter duplicate tool calls
      const filteredContent = msg.content.filter((item, contentIndex) => {
        const key = `${msgIndex}:${contentIndex}`;
        if (duplicateOccurrences.has(key)) {
          return false; // Remove this duplicate
        }
        return true;
      });
      
      if (filteredContent.length !== msg.content.length) {
        if (isVerboseDebugEnabled()) {
          console.error(`[ANYCLAUDE DEBUG] Filtered ${msg.content.length - filteredContent.length} duplicate tool calls from assistant message ${msgIndex}`);
        }
        return { ...msg, content: filteredContent };
      }
    } else if (msg.role === 'user' && Array.isArray(msg.content)) {
      // Filter duplicate tool results
      const filteredContent = msg.content.filter((item, contentIndex) => {
        const key = `${msgIndex}:${contentIndex}`;
        if (duplicateResults.has(key)) {
          return false; // Remove this duplicate
        }
        return true;
      });
      
      if (filteredContent.length !== msg.content.length) {
        if (isVerboseDebugEnabled()) {
          console.error(`[ANYCLAUDE DEBUG] Filtered ${msg.content.length - filteredContent.length} duplicate tool results from user message ${msgIndex}`);
        }
        return { ...msg, content: filteredContent };
      }
    }
    
    return msg;
  });

  // Step 4: Verify consistency and order preservation
  const validation = validateToolCallConsistency(filteredMessages);
  
  if (isVerboseDebugEnabled()) {
    console.error(`[ANYCLAUDE DEBUG] Final state: ${validation.toolCalls.size} tool calls, ${validation.toolResults.size} tool results`);
    
    // Check for orphaned results (this is a critical error)
    validation.orphanedResults.forEach(resultId => {
      console.error(`[ANYCLAUDE DEBUG] ERROR: Tool result '${resultId}' has no corresponding tool call - this will cause API errors!`);
    });
    
    // Check for calls without results (this is OK in streaming scenarios)
    validation.callsWithoutResults.forEach(callId => {
      console.error(`[ANYCLAUDE DEBUG] INFO: Tool call '${callId}' has no result (may be in progress)`);
    });
    
    // Verify order preservation
    if (validation.orderPreserved) {
      console.error(`[ANYCLAUDE DEBUG] Order preservation: OK - tool calls and results maintain their relative order`);
    } else {
      console.error(`[ANYCLAUDE DEBUG] WARNING: Order may not be preserved between tool calls and results`);
    }
  } else if (isDebugEnabled() && validation.orphanedResults.size > 0) {
    // Always show critical errors even at level 1
    console.error(`[ANYCLAUDE DEBUG] ERROR: ${validation.orphanedResults.size} orphaned tool results detected - this will cause API errors`);
  }
  
  // Throw an error if we have orphaned results (critical issue)
  if (validation.orphanedResults.size > 0) {
    throw new Error(
      `Filter created orphaned tool results without corresponding calls: ${Array.from(validation.orphanedResults).join(', ')}. ` +
      `This will cause API errors. Please report this bug.`
    );
  }

  return filteredMessages;
}

/**
 * Validates that tool calls and results are properly paired and ordered.
 * Returns detailed information about the consistency of the filtered messages.
 */
function validateToolCallConsistency(messages: AnthropicMessage[]) {
  const toolCalls = new Set<string>();
  const toolResults = new Set<string>();
  const toolCallOrder: string[] = [];
  const toolResultOrder: string[] = [];
  
  messages.forEach(msg => {
    if (msg.role === 'assistant' && Array.isArray(msg.content)) {
      msg.content.forEach(item => {
        if (typeof item === 'object' && item.type === 'tool_use') {
          toolCalls.add(item.id);
          toolCallOrder.push(item.id);
        }
      });
    } else if (msg.role === 'user' && Array.isArray(msg.content)) {
      msg.content.forEach(item => {
        if (typeof item === 'object' && item.type === 'tool_result') {
          toolResults.add(item.tool_use_id);
          toolResultOrder.push(item.tool_use_id);
        }
      });
    }
  });
  
  // Find orphaned results (results without corresponding calls)
  const orphanedResults = new Set<string>();
  toolResults.forEach(resultId => {
    if (!toolCalls.has(resultId)) {
      orphanedResults.add(resultId);
    }
  });
  
  // Find calls without results
  const callsWithoutResults = new Set<string>();
  toolCalls.forEach(callId => {
    if (!toolResults.has(callId)) {
      callsWithoutResults.add(callId);
    }
  });
  
  // Check if order is preserved (tool results should appear in same order as calls)
  // This is a simplified check - we verify that for IDs that appear in both lists,
  // their relative order is maintained
  const commonIds = toolCallOrder.filter(id => toolResults.has(id));
  const resultOrderForCommon = toolResultOrder.filter(id => toolCalls.has(id));
  const orderPreserved = JSON.stringify(commonIds) === JSON.stringify(resultOrderForCommon);
  
  return {
    toolCalls,
    toolResults,
    orphanedResults,
    callsWithoutResults,
    orderPreserved,
    toolCallOrder,
    toolResultOrder,
  };
}