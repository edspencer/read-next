import { ReadNext, defaultSummarizationPrompt } from "../ReadNext";
import type { SaveableVectorStore, VectorStore } from "@langchain/core/vectorstores";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";

import fs from "fs";
import winston from "winston";

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import path from "path";

import { FakeChatModel, FakeEmbeddings, FakeVectorStore } from "@langchain/core/utils/testing";

class SaveableFakeVectorStore extends FakeVectorStore implements SaveableVectorStore {
  async save(): Promise<void> {
    return;
  }
}

const pageContent = fs.readFileSync(path.join(__dirname, "document.mdx"), "utf-8");

describe("ReadNext", () => {
  describe("creating an instance", () => {
    let unconfigured: ReadNext;
    let configured: ReadNext;

    const cacheDir = "/tmp/read-next-test";

    beforeEach(async () => {
      unconfigured = await ReadNext.create();

      configured = await ReadNext.create({
        cacheDir,
        summaryModel: new FakeChatModel({}),
        embeddingsModel: new FakeEmbeddings({}),
      });
    });

    it("accepts a custom embedding model", () => {
      expect(configured.embeddingsModel).toBeInstanceOf(FakeEmbeddings);
    });

    it("supplies a default embedding model", () => {
      expect(unconfigured.embeddingsModel).toBeInstanceOf(OpenAIEmbeddings);
    });

    it("accepts a custom summary model", () => {
      expect(configured.summaryModel).toBeInstanceOf(FakeChatModel);
    });

    it("supplies a default summary model", () => {
      expect(unconfigured.summaryModel).toBeInstanceOf(ChatOpenAI);
    });

    it("accepts a cacheDir path", () => {
      expect(configured.cacheDir).toBe(cacheDir);
    });

    it("accepts the absence of a cacheDir path, in which case it will use a tmp dir", () => {
      expect(typeof unconfigured.cacheDir).toBe("string");
    });

    it("supplies a default FAISS vectorStore with the supplied embedding model", async () => {
      const instance = await ReadNext.create({
        embeddingsModel: new FakeEmbeddings({}),
      });
      expect(instance.vectorStore).toBeInstanceOf(FaissStore);
    });

    it("supplies a default FAISS vectorStore with a default embedding model", async () => {
      const instance = await ReadNext.create();
      expect(instance.vectorStore).toBeInstanceOf(FaissStore);
    });

    it("accepts a custom vectorStore", async () => {
      const customVectorStore = new SaveableFakeVectorStore(new FakeEmbeddings({}));
      const instance = await ReadNext.create({
        vectorStore: customVectorStore,
      });
      expect(instance.vectorStore).toBe(customVectorStore);
    });

    it("accepts a custom summarization prompt", async () => {
      const customPrompt = "Custom summarization prompt";
      const instance = await ReadNext.create({
        summarizationPrompt: customPrompt,
      });
      expect(instance.summarizationPrompt).toBe(customPrompt);
    });

    it("supplies a default summarization prompt if none is provided", async () => {
      const instance = await ReadNext.create();
      expect(instance.summarizationPrompt).toBe(defaultSummarizationPrompt);
    });

    it("accepts a custom logger", async () => {
      const customLogger = winston.createLogger();
      const instance = await ReadNext.create({
        logger: customLogger,
      });
      expect(instance.logger).toBe(customLogger);
    });

    it("supplies a default logger if none is provided", async () => {
      const instance = await ReadNext.create();
      expect(instance.logger).toBeInstanceOf(winston.Logger);
    });
  });

  const logger = winston.createLogger({
    silent: true,
  });

  describe("indexing documents", () => {
    let engine: ReadNext;
    let summaryModel: BaseChatModel;
    let vectorStore: SaveableFakeVectorStore;
    let embeddingsModel: FakeEmbeddings;

    const sourceDocument = { pageContent: "hello world", id: "1", metadata: {} };

    beforeEach(async () => {
      summaryModel = new FakeChatModel({});
      embeddingsModel = new FakeEmbeddings({});
      vectorStore = new SaveableFakeVectorStore(embeddingsModel);

      engine = await ReadNext.create({
        summaryModel,
        vectorStore,
        logger,
      });
    });

    it("generates a summary for each document", async () => {
      const getSummaryForSpy = jest.spyOn(engine, "getSummaryFor");

      await engine.index({ sourceDocuments: [sourceDocument] });

      expect(getSummaryForSpy).toHaveBeenCalled();
      expect(getSummaryForSpy).toHaveBeenCalledWith({ sourceDocument });
    });

    it("adds the summary documents to the vectorStore (including the generated summary)", async () => {
      const sourceDocument = {
        pageContent,
        id: "1",
      };

      const fakeSummary = "fake summary";
      jest.spyOn(engine, "getSummaryFor").mockResolvedValue(fakeSummary);

      const addDocumentsSpy = jest.spyOn(vectorStore, "addDocuments");

      await engine.index({ sourceDocuments: [sourceDocument] });

      expect(addDocumentsSpy).toHaveBeenCalled();
      expect(addDocumentsSpy).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            pageContent: fakeSummary,
            metadata: expect.objectContaining({
              sourceDocumentId: sourceDocument.id,
            }),
          }),
        ]),
        expect.objectContaining({
          ids: expect.arrayContaining([sourceDocument.id]),
        })
      );
    });

    it("saves the vectorStore once", async () => {
      const saveSpy = jest.spyOn(vectorStore, "save");

      await engine.index({ sourceDocuments: [sourceDocument] });

      expect(saveSpy).toHaveBeenCalledTimes(1);
    });

    it("returns the summary documents", async () => {
      const summaries = await engine.index({ sourceDocuments: [sourceDocument] });

      expect(summaries).toHaveLength(1);
      expect(summaries[0]).toHaveProperty("pageContent");
      expect(summaries[0]).toHaveProperty("metadata.sourceDocumentId", sourceDocument.id);
    });

    it("adds the document to the content hasher", async () => {
      const setSpy = jest.spyOn(engine.contentHasher, "set");

      await engine.index({ sourceDocuments: [sourceDocument] });

      expect(setSpy).toHaveBeenCalledWith(sourceDocument);
    });

    it("saves the content hasher after each document is indexed", async () => {
      const saveSpy = jest.spyOn(engine.contentHasher, "save");

      await engine.index({ sourceDocuments: [sourceDocument] });

      expect(saveSpy).toHaveBeenCalled();
    });
  });

  describe("getSummaryFor", () => {
    let engine: ReadNext;
    let summaryModel: BaseChatModel;
    let vectorStore: SaveableFakeVectorStore;
    let embeddingsModel: FakeEmbeddings;

    const sourceDocument = { pageContent: "hello world", id: "1", metadata: {} };
    const fakeSummary = "fake summary";

    beforeEach(async () => {
      summaryModel = new FakeChatModel({});
      embeddingsModel = new FakeEmbeddings({});
      vectorStore = new SaveableFakeVectorStore(embeddingsModel);

      engine = await ReadNext.create({
        summaryModel,
        vectorStore,
        logger,
      });

      jest.spyOn(engine.contentHasher, "set");
      jest.spyOn(engine.contentHasher, "save");
    });

    it("generates a new summary if one does not exist", async () => {
      jest.spyOn(engine.contentHasher, "hasFresh").mockReturnValue(false);
      jest.spyOn(engine, "summarize").mockResolvedValue(fakeSummary);

      const summary = await engine.getSummaryFor({ sourceDocument });

      expect(summary).toBe(fakeSummary);
      expect(engine.summarize).toHaveBeenCalledWith({ sourceDocument });
      expect(engine.contentHasher.set).toHaveBeenCalledWith(sourceDocument);
      expect(engine.contentHasher.save).toHaveBeenCalled();
    });

    it("generates a new summary if the content has changed since last time", async () => {
      jest.spyOn(engine.contentHasher, "hasFresh").mockReturnValue(false);
      jest.spyOn(engine, "summarize").mockResolvedValue(fakeSummary);

      const summary = await engine.getSummaryFor({ sourceDocument });

      expect(summary).toBe(fakeSummary);
      expect(engine.summarize).toHaveBeenCalledWith({ sourceDocument });
      expect(engine.contentHasher.set).toHaveBeenCalledWith(sourceDocument);
      expect(engine.contentHasher.save).toHaveBeenCalled();
    });

    it("returns the existing summary if it exists and the content has not changed", async () => {
      jest.spyOn(engine, "summarize").mockResolvedValue(fakeSummary);

      const summary = await engine.getSummaryFor({ sourceDocument });

      expect(summary).toBe(fakeSummary);
      expect(engine.summarize).not.toHaveBeenCalled();
      expect(engine.contentHasher.set).not.toHaveBeenCalled();
      expect(engine.contentHasher.save).not.toHaveBeenCalled();
    });
  });
});
