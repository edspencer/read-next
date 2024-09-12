import { createHash } from "crypto";
import fs from "fs";
import path from "path";
import winston from "winston";
import ContentHasher from "../ContentHasher";
import type { DocumentInput } from "@langchain/core/documents";

describe("ContentHasher", () => {
  let contentHasher: ContentHasher;
  let logger: winston.Logger;
  const cacheDir = "/tmp/content-hasher-test";
  const cacheFile = path.join(cacheDir, "contentHashes.json");

  beforeEach(() => {
    logger = winston.createLogger({
      silent: true,
    });

    if (!fs.existsSync(cacheDir)) {
      fs.mkdirSync(cacheDir);
    }

    contentHasher = new ContentHasher({ cacheDir, logger });
  });

  afterEach(() => {
    if (fs.existsSync(cacheFile)) {
      fs.unlinkSync(cacheFile);
    }
  });

  describe("constructor", () => {
    it("initializes with an empty records map", () => {
      expect(contentHasher.records.size).toBe(0);
    });

    it("sets the cacheDir and cacheFile properties", () => {
      expect(contentHasher.cacheDir).toBe(cacheDir);
      expect(contentHasher.cacheFile).toBe(cacheFile);
    });

    it("loads existing content hashes from the cache file", () => {
      const records = { "1": "hash1" };
      fs.writeFileSync(cacheFile, JSON.stringify(records));

      const newContentHasher = new ContentHasher({ cacheDir, logger });
      expect(newContentHasher.records.size).toBe(1);
      expect(newContentHasher.records.get("1")).toBe("hash1");
    });
  });

  describe("hasFresh", () => {
    it("returns true if the document's content is fresh", () => {
      const document: DocumentInput = { pageContent: "hello world", id: "1" };
      const contentSha = createHash("sha256").update(document.pageContent).digest("hex");
      contentHasher.records.set(String(document.id), contentSha);

      expect(contentHasher.hasFresh(document)).toBe(true);
    });

    it("returns false if the document's content is not fresh", () => {
      const document: DocumentInput = { pageContent: "hello world", id: "1" };
      contentHasher.records.set(String(document.id), "differentHash");

      expect(contentHasher.hasFresh(document)).toBe(false);
    });

    it("returns false and logs a warning if the document has no id", () => {
      const document: DocumentInput = { pageContent: "hello world" };
      const warnSpy = jest.spyOn(logger, "warn");

      expect(contentHasher.hasFresh(document)).toBe(false);
      expect(warnSpy).toHaveBeenCalledWith("No id supplied, so caching will not work");
    });
  });

  describe("set", () => {
    it("sets the content hash for a given document", () => {
      const document: DocumentInput = { pageContent: "hello world", id: "1" };
      const contentSha = createHash("sha256").update(document.pageContent).digest("hex");

      contentHasher.set(document);
      expect(contentHasher.records.get(String(document.id))).toBe(contentSha);
    });

    it("logs a warning if the document has no id", () => {
      const document: DocumentInput = { pageContent: "hello world" };
      const warnSpy = jest.spyOn(logger, "warn");

      contentHasher.set(document);
      expect(warnSpy).toHaveBeenCalledWith("No id supplied, so caching will not work");
    });
  });

  describe("load", () => {
    it("loads content hashes from the cache file", () => {
      const records = { "1": "hash1" };
      fs.writeFileSync(cacheFile, JSON.stringify(records));

      expect(contentHasher.load()).toBe(true);
      expect(contentHasher.records.size).toBe(1);
      expect(contentHasher.records.get("1")).toBe("hash1");
    });

    it("returns false and logs an error if loading fails", () => {
      const errorSpy = jest.spyOn(logger, "error");
      fs.writeFileSync(cacheFile, "invalid json");

      expect(contentHasher.load()).toBe(false);
      expect(errorSpy).toHaveBeenCalledWith("Error loading content hashes");
    });
  });

  describe("save", () => {
    it("saves the current content hashes to the cache file", () => {
      const document: DocumentInput = { pageContent: "hello world", id: "1" };
      contentHasher.set(document);

      contentHasher.save();
      const savedRecords = JSON.parse(fs.readFileSync(cacheFile, "utf-8"));
      expect(savedRecords["1"]).toBe(contentHasher.records.get("1"));
    });
  });
});
