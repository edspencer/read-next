import { createHash } from "crypto";
import fs from "fs";
import path from "path";
import winston from "winston";
import type { DocumentInput } from "@langchain/core/documents";

/**
 * The `ContentHasher` class is responsible for managing content hashes for documents.
 * It provides methods to check if a document's content is fresh, set new content hashes,
 * and load/save these hashes from/to a cache file.
 */
export default class ContentHasher {
  /**
   * A map that stores document IDs and their corresponding content hashes.
   */
  records: Map<string, string>;

  /**
   * The directory where the cache file is stored.
   */
  cacheDir: string;

  /**
   * The path to the cache file where content hashes are stored.
   */
  cacheFile: string;

  /**
   * A logger instance for logging messages and errors.
   */
  logger: winston.Logger;

  /**
   * Constructs a new `ContentHasher` instance.
   *
   * @param cacheDir - The directory where the cache file is stored.
   * @param logger - A logger instance for logging messages and errors.
   */
  constructor({ cacheDir, logger }: { cacheDir: string; logger: winston.Logger }) {
    this.records = new Map();
    this.logger = logger;

    this.cacheDir = cacheDir;
    this.cacheFile = path.join(cacheDir, "contentHashes.json");
    this.load();
  }

  /**
   * Checks if the content of a given document is fresh by comparing its hash with the stored hash.
   *
   * @param document - The document to check.
   * @returns `true` if the document's content is fresh, `false` otherwise.
   */
  hasFresh(document: DocumentInput): boolean {
    const { pageContent, id } = document;

    const contentSha = createHash("sha256").update(pageContent).digest("hex");

    if (id) {
      return this.records.get(id) === contentSha;
    } else {
      this.logger.warn("No id supplied, so caching will not work");
      return false;
    }
  }

  /**
   * Sets the content hash for a given document.
   *
   * @param document - The document to set the hash for.
   */
  set(document: DocumentInput) {
    const { pageContent, id } = document;

    const contentSha = createHash("sha256").update(pageContent).digest("hex");

    if (id) {
      this.records.set(id, contentSha);
    } else {
      this.logger.warn("No id supplied, so caching will not work");
    }
  }

  /**
   * Loads the content hashes from the cache file.
   *
   * @returns `true` if the content hashes were successfully loaded, `false` otherwise.
   */
  load(): boolean {
    try {
      if (fs.existsSync(this.cacheFile)) {
        const records = JSON.parse(fs.readFileSync(this.cacheFile, "utf-8"));
        this.records = new Map(Object.entries(records));
      }
    } catch (e) {
      this.logger.error("Error loading content hashes");
      this.logger.error(e);

      return false;
    }

    return true;
  }

  /**
   * Saves the current content hashes to the cache file.
   */
  save(): void {
    fs.writeFileSync(this.cacheFile, JSON.stringify(Object.fromEntries(this.records), null, 2));
  }
}
