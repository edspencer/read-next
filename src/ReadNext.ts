import type { Document, DocumentInput } from "@langchain/core/documents";
import type { VectorStore } from "@langchain/core/vectorstores";
import type { Embeddings, EmbeddingsInterface } from "@langchain/core/embeddings";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";

import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Runnable } from "@langchain/core/runnables";

import { RecordManager } from "@langchain/core/indexing";
import winston from "winston";

export type DocumentID = string | number;

export type Embedding = number[];

export type SummaryEmbedding = {
  _id: DocumentID;
  sourceDocumentId?: DocumentID;
  $vector: Embedding;
  content: string;
};

interface ReadNextArgs {
  vectorStore: VectorStore;
  summaryModel: BaseChatModel;
  summarizationPrompt?: string;
  cacheSummaries?: boolean;
  cacheDir?: string;
}

interface CreateReadNextArgs {
  vectorStore?: VectorStore;
  embeddingsModel?: Embeddings;
  recordManager?: RecordManager;
  summaryModel?: BaseChatModel;
  summarizationPrompt?: string;
  cacheSummaries?: boolean;
  cacheDir?: string;
}

interface Summarize {
  sourceDocument: DocumentInput;
  saveSummary?: boolean;
}

interface Suggest {
  sourceDocument: DocumentInput;
  limit?: number;
  ignore?: Document[];
}

const defaultSummarizationPrompt = `Here is an article for you to summarize.
  The purpose of the summarization is to drive recommendations for what article somebody should read next,
  based on the article they are currently reading. The summarization should be as lengthy as necessary to
  capture the full essence of the article. It is not intended to be a short summary, but more of a condensing of the article.
  All of the summarizations will be passed through an embedding model, with the embeddings used to rank the articles.

  Please do not reply with any text other than the summary.`;

import fs from "fs";
import path from "path";
import os from "os";
import { FaissStore } from "@langchain/community/vectorstores/faiss";

import { createHash } from "crypto";

export class ReadNext {
  summaryModel: BaseChatModel;
  embeddingsModel: EmbeddingsInterface;
  summarizationPrompt: string;
  vectorStore: VectorStore;

  summaryParser: StringOutputParser;
  summaryChain: Runnable;

  cacheSummaries: boolean;
  cacheDir: string;

  contentHasher: ContentHasher;

  logger: winston.Logger;

  static async create(config: CreateReadNextArgs = {}): Promise<ReadNext> {
    let { cacheSummaries = true, cacheDir, vectorStore, recordManager, summaryModel } = config;

    if (!config.embeddingsModel) {
      config.embeddingsModel = new OpenAIEmbeddings({ model: "text-embedding-ada-002" });
    }

    if (!vectorStore) {
      if (cacheSummaries && cacheDir) {
        const faissPath = path.join(cacheDir, "faiss.index");
        if (fs.existsSync(faissPath)) {
          vectorStore = await FaissStore.load(cacheDir, config.embeddingsModel);
        }
      }
    }

    if (!vectorStore) {
      vectorStore = new FaissStore(config.embeddingsModel, {});
    }

    if (!summaryModel) {
      summaryModel = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
    }

    const readNextConfig: ReadNextArgs = {
      vectorStore,
      summaryModel,
      cacheDir,
    };

    return new ReadNext(readNextConfig);
  }

  constructor({
    vectorStore,
    summaryModel,
    cacheSummaries = false,
    summarizationPrompt = defaultSummarizationPrompt,
    cacheDir,
  }: ReadNextArgs) {
    this.vectorStore = vectorStore;
    this.summaryModel = summaryModel;
    this.summarizationPrompt = summarizationPrompt;
    this.embeddingsModel = this.vectorStore.embeddings;

    this.cacheSummaries = cacheSummaries;

    this.summaryParser = new StringOutputParser();
    this.summaryChain = this.summaryModel.pipe(this.summaryParser);

    this.cacheDir = cacheDir || os.tmpdir();

    this.logger = winston.createLogger({
      level: "info",
      transports: [
        //
        // - Write all logs with importance level of `error` or less to `error.log`
        // - Write all logs with importance level of `info` or less to `combined.log`
        //
        new winston.transports.File({
          format: winston.format.simple(),
          filename: path.join(this.cacheDir, "readnext-error.log"),
          level: "error",
        }),
        new winston.transports.File({
          format: winston.format.simple(),
          filename: path.join(this.cacheDir, "readnext.log"),
        }),
        new winston.transports.Console({ format: winston.format.cli() }),
      ],
    });

    this.contentHasher = new ContentHasher({ cacheDir: this.cacheDir, logger: this.logger });
  }

  async index({ sourceDocuments, force = false }: { sourceDocuments: DocumentInput[]; force?: boolean }) {
    const summaryDocuments: Document[] = [];
    let embeddingsAdded = 0;

    for (const sourceDocument of sourceDocuments) {
      //create the summary
      const summary = await this.getSummaryFor({ sourceDocument });

      const summaryDocument: Document = {
        pageContent: summary,
        metadata: {
          sourceDocumentId: sourceDocument.id,
        },
      };

      summaryDocuments.push(summaryDocument);

      this.logger.info(`Adding ${sourceDocument.id} to the vector store`);
      try {
        await this.vectorStore.delete({ ids: [sourceDocument.id] });
      } catch (e) {
        // ignore errors here as we don't care if the document doesn't already exist
      }
      await this.vectorStore.addDocuments([summaryDocument], { ids: [sourceDocument.id] });
      embeddingsAdded += 1;

      this.logger.info(`${sourceDocument.id} added to the vector store`);

      //save to the cache
      this.contentHasher.set(sourceDocument);
      this.contentHasher.save();
    }

    if (embeddingsAdded > 0) {
      //The store throws an error if .addDocuments is never called, so we only save if we have added documents
      try {
        // @ts-ignore
        if (typeof this.vectorStore.save === "function") {
          // @ts-ignore
          await this.vectorStore.save(this.cacheDir);
        }
      } catch (e) {
        console.error("Error saving vector store");
        console.error(e);
      }
    }

    return summaryDocuments;
  }

  async getSummaryFor({ sourceDocument }: { sourceDocument: DocumentInput }) {
    let summary: string | undefined = undefined;
    const hasId = sourceDocument.id;
    const hasFresh = this.contentHasher.hasFresh(sourceDocument);

    const summariesDir = path.join(this.cacheDir, "summaries");
    const summaryFileName = path.join(summariesDir, `${sourceDocument.id}`);

    fs.mkdirSync(summariesDir, { recursive: true });

    if (hasId) {
      if (hasFresh) {
        if (fs.existsSync(summaryFileName)) {
          summary = fs.readFileSync(summaryFileName, "utf8");
        }
      }
    } else {
      console.warn("No id found for document, summary will not be cached");
    }

    if (summary === undefined) {
      summary = await this.summarize({ sourceDocument });
    } else {
      this.logger.info(`Using cached summary for ${sourceDocument.id}`);
    }

    if (hasId) {
      fs.writeFileSync(summaryFileName, summary);
    }

    //save the new content sha
    if (!hasFresh) {
      this.contentHasher.set(sourceDocument);
      this.contentHasher.save();
    }

    return summary;
  }

  /**
   * Summarizes a source document, optionally saving the summary to a directory of your choice.
   * Creates an embedding for the summary and saves it to the vector store.
   * Returns the Summary
   * @returns
   */
  async summarize({ sourceDocument }: Summarize): Promise<string> {
    const { pageContent } = sourceDocument;

    const messages = [new SystemMessage(this.summarizationPrompt), new HumanMessage(pageContent)];

    this.logger.info(`Generating summary for ${sourceDocument.id}`);
    const summary = await this.summaryChain.invoke(messages);
    this.logger.info("Summarization completed");

    return summary;
  }

  async suggest({ sourceDocument, limit = 10 }: Suggest): Promise<Suggestions> {
    this.logger.info(`Getting suggestion for ${sourceDocument.id}`);
    const summary = await this.getSummaryFor({ sourceDocument });
    const results = await this.vectorStore.similaritySearchWithScore(summary, limit + 1);

    const related = results
      .filter(([doc]) => doc.metadata.sourceDocumentId !== sourceDocument.id)
      .map(
        ([
          {
            metadata: { sourceDocumentId },
          },
          score,
        ]) => ({ sourceDocumentId, score })
      );

    return {
      id: sourceDocument.id,
      related: related.slice(0, limit + 1),
    };
  }
}

type RelatedDocument = {
  sourceDocumentId: string;
  score: number;
};

type Suggestions = {
  id?: string;
  related: RelatedDocument[];
};

class ContentHasher {
  records: Map<string, string>;
  cacheFile: string;
  logger: winston.Logger;

  constructor({ cacheDir, logger }: { cacheDir: string; logger: winston.Logger }) {
    this.records = new Map();
    this.logger = logger;

    this.cacheFile = path.join(cacheDir, "contentHashes.json");
    this.load();
  }

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

  set(document: DocumentInput) {
    const { pageContent, id } = document;

    const contentSha = createHash("sha256").update(pageContent).digest("hex");

    if (id) {
      this.records.set(id, contentSha);
    } else {
      this.logger.warn("No id supplied, so caching will not work");
    }
  }

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

  save(): void {
    fs.writeFileSync(this.cacheFile, JSON.stringify(Object.fromEntries(this.records), null, 2));
  }
}
