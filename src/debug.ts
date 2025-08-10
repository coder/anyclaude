import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// Store error messages to display later
let pendingErrorMessages: string[] = [];

export interface DebugInfo {
  statusCode: number;
  request: {
    method?: string;
    url?: string;
    headers: any;
    body: any;
  };
  response?: {
    statusCode: number;
    headers: any;
    body: string;
  };
}

/**
 * Write debug info to a temp file when ANYCLAUDE_DEBUG is set
 * @returns The path to the debug file, or null if not written
 */
export function writeDebugToTempFile(
  statusCode: number,
  request: DebugInfo["request"],
  response?: DebugInfo["response"]
): string | null {
  // Log 4xx errors (except 429) when ANYCLAUDE_DEBUG is set
  const debugEnabled = process.env.ANYCLAUDE_DEBUG;
  
  if (!debugEnabled || statusCode === 429 || statusCode < 400 || statusCode >= 500) {
    return null;
  }

  try {
    const tmpDir = os.tmpdir();
    const timestamp = Date.now();
    const randomId = Math.random().toString(36).substring(2, 8);
    const filename = `anyclaude-debug-${timestamp}-${randomId}.json`;
    const filepath = path.join(tmpDir, filename);

    const debugData = {
      timestamp: new Date().toISOString(),
      request: {
        method: request.method,
        url: request.url,
        headers: request.headers,
        body: request.body,
      },
      response: response || null,
    };

    fs.writeFileSync(filepath, JSON.stringify(debugData, null, 2), 'utf8');
    
    // Also write a simpler error log file that's easier to tail
    const errorLogPath = path.join(tmpDir, 'anyclaude-errors.log');
    const errorMessage = `[${new Date().toISOString()}] HTTP ${statusCode} - Debug: ${filepath}\n`;
    fs.appendFileSync(errorLogPath, errorMessage, 'utf8');
    
    return filepath;
  } catch (error) {
    console.error("[ANYCLAUDE DEBUG] Failed to write debug file:", error);
    return null;
  }
}

/**
 * Queue an error message to be displayed later
 */
export function queueErrorMessage(message: string): void {
  pendingErrorMessages.push(message);
  // Display after a short delay to avoid being overwritten
  setTimeout(displayPendingErrors, 500);
}

/**
 * Display pending error messages to stderr with formatting
 */
function displayPendingErrors(): void {
  if (pendingErrorMessages.length > 0) {
    // Use stderr and add newlines to separate from Claude's output
    process.stderr.write('\n\n═══════════════════════════════════════\n');
    process.stderr.write('ANYCLAUDE DEBUG - Errors detected:\n');
    process.stderr.write('═══════════════════════════════════════\n');
    pendingErrorMessages.forEach(msg => {
      process.stderr.write(msg + '\n');
    });
    process.stderr.write('═══════════════════════════════════════\n\n');
    pendingErrorMessages = [];
  }
}

/**
 * Log a debug error and queue it for display
 */
export function logDebugError(
  type: "HTTP" | "Provider" | "Streaming",
  statusCode: number,
  debugFile: string | null,
  context?: { provider?: string; model?: string }
): void {
  if (!debugFile) return;
  
  let message = `${type} error`;
  if (context?.provider && context?.model) {
    message += ` (${context.provider}/${context.model})`;
  } else if (statusCode) {
    message += ` ${statusCode}`;
  }
  message += ` - Debug info written to: ${debugFile}`;
  
  queueErrorMessage(message);
}

/**
 * Display debug mode startup message
 */
export function displayDebugStartup(): void {
  if (process.env.ANYCLAUDE_DEBUG) {
    const tmpDir = os.tmpdir();
    const errorLogPath = path.join(tmpDir, 'anyclaude-errors.log');
    process.stderr.write('\n═══════════════════════════════════════\n');
    process.stderr.write('ANYCLAUDE DEBUG MODE ENABLED\n');
    process.stderr.write(`Error log: ${errorLogPath}\n`);
    process.stderr.write(`Debug files: ${tmpDir}/anyclaude-debug-*.json\n`);
    process.stderr.write('═══════════════════════════════════════\n\n');
  }
}

/**
 * Check if debug mode is enabled
 */
export function isDebugEnabled(): boolean {
  return !!process.env.ANYCLAUDE_DEBUG;
}