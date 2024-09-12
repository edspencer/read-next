/**
 * Custom Winston formatter to indicate cache hits/misses and expensive operations.
 *
 * This formatter uses specific symbols to visually represent the status of cache operations:
 * - `cacheHit`: Represented by a green dot (•).
 * - `cacheMiss`: Represented by a red dot (•).
 * - `expensive`: Represented by a yellow hourglass (⏳).
 *
 * The formatter combines colorization, simple formatting, and custom message formatting
 * to prepend the appropriate symbols to log messages based on the `expensive` and `cache` properties.
 *
 * @example
 * ```typescript
 * import { readNextLogger } from './logger';
 *
 * readNextLogger.info('This is a cache hit message', { cache: 'hit' });
 * readNextLogger.info('This is a cache miss message', { cache: 'miss' });
 * readNextLogger.info('This is an expensive operation message', { expensive: true });
 * ```
 *
 * @constant
 * @type {winston.Logform.Format}
 */
import winston from "winston";

const cacheHit = "\u001b[32m•\u001b[0m";
const cacheMiss = "\u001b[31m•\u001b[0m";
const expensive = "\u001b[33m⏳\u001b[0m";

//tracks the most recent log's id metadata to decide if we should start a new group
let previousId: string | undefined = undefined;

//simple custom winston formatter to indicate cache hits/misses and expensive operations
export const cacheIndicatingLogFormatter = winston.format.combine(
  winston.format.colorize(),
  winston.format.simple(),
  winston.format.printf((info) => {
    let { message } = info;
    let labels = [];
    let newGroup = null;

    if (info.id !== previousId) {
      previousId = info.id;
      newGroup = `\n${info.id}\n`;
    }

    if (info.expensive) {
      //null as a second element because expensive is a wide character
      labels.push(expensive, null);
    }

    if (info.cache === "hit") {
      labels.push(cacheHit);
    }

    if (info.cache === "miss") {
      labels.push(cacheMiss);
    }

    while (labels.length < 4) {
      labels.push(" ");
    }

    return [newGroup, labels.join(""), message].join("");
  })
);

/**
 * Custom Winston logger for read-next operations.
 */
export const readNextLogger = winston.createLogger({
  level: "info",
  transports: [new winston.transports.Console({ format: cacheIndicatingLogFormatter })],
});
