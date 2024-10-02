import fs from "fs";
import path from "path";
import os from "os";

import type { Document, DocumentInput } from "@langchain/core/documents";
import type { VectorStore } from "@langchain/core/vectorstores";
import type { Embeddings, EmbeddingsInterface } from "@langchain/core/embeddings";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { FaissStore } from "@langchain/community/vectorstores/faiss";

import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Runnable } from "@langchain/core/runnables";

import winston from "winston";

import { readNextLogger } from "./logger";
import ContentHasher from "./ContentHasher";

interface ReadNextArgs {
  vectorStore: VectorStore;
  summaryModel: BaseChatModel;
  summarizationPrompt?: SummarizationPrompt;
  cacheDir?: string;
  logger?: winston.Logger;
  parallel?: number;
  getSourceDocument?: (doc: any) => DocumentInput;
  sourceDocuments?: any[];
}

interface CreateReadNextArgs {
  vectorStore?: VectorStore;
  embeddingsModel?: Embeddings;
  summaryModel?: BaseChatModel;
  summarizationPrompt?: SummarizationPrompt;
  cacheDir?: string;
  logger?: winston.Logger;
  parallel?: number;
  getSourceDocument?: (doc: any) => DocumentInput;
  sourceDocuments?: any[];
}

interface Summarize {
  sourceDocument: DocumentInput;
  saveSummary?: boolean;
  summarizationPrompt?: SummarizationPrompt;
}

interface Suggest {
  sourceDocument: DocumentInput | any;
  limit?: number;
  ignore?: Document[];
  includeSummary?: boolean;
}

type SummarizationPrompt = string | ((doc: DocumentInput) => string);

interface GetSummaryArgs {
  sourceDocument: DocumentInput;
  summarizationPrompt?: SummarizationPrompt;
}

interface IndexArgs {
  sourceDocuments: DocumentInput[];
  parallel?: number;
  summarizationPrompt?: SummarizationPrompt;
}

export interface PromptArgs {
  sourceDocuments?: any[];
  prompt: any;
  docId?: string;
}

export const defaultSummarizationPrompt = `Here is an article for you to summarize.
  The purpose of the summarization is to drive recommendations for what article somebody should read next,
  based on the article they are currently reading. The summarization should be as lengthy as necessary to
  capture the full essence of the article. It is not intended to be a short summary, but more of a condensing of the article.
  All of the summarizations will be passed through an embedding model, with the embeddings used to rank the articles.

  Please do not reply with any text other than the summary.`;

/**
 * The `ReadNext` class provides functionality for summarizing documents, creating embeddings,
 * and storing them in a vector store for similarity searches. It also supports caching summaries
 * and embeddings to improve performance.
 *
 * @class
 * @property {BaseChatModel} summaryModel - The model used for generating summaries.
 * @property {EmbeddingsInterface} embeddingsModel - The model used for generating embeddings.
 * @property {string} summarizationPrompt - The prompt used for summarization.
 * @property {VectorStore} vectorStore - The store used for storing vector embeddings.
 * @property {StringOutputParser} summaryParser - The parser used for parsing summary outputs.
 * @property {Runnable} summaryChain - The chain of operations for summarization.
 * @property {string} cacheDir - The directory used for caching summaries and embeddings.
 * @property {ContentHasher} contentHasher - The hasher used for content hashing.
 * @property {winston.Logger} logger - The logger used for logging information.
 *
 */
export class ReadNext {
  summaryModel: BaseChatModel;
  embeddingsModel: EmbeddingsInterface;
  summarizationPrompt: SummarizationPrompt;
  vectorStore: VectorStore;

  summaryParser: StringOutputParser;
  summaryChain: Runnable;

  cacheDir: string;

  contentHasher: ContentHasher;

  logger: winston.Logger;
  parallel: number;
  getSourceDocument: (doc: any) => DocumentInput = (doc) => doc;
  sourceDocuments: any[];

  /**
   * Creates an instance of ReadNext with the provided configuration.
   *
   * @param {CreateReadNextArgs} [config={}] - The configuration object for creating ReadNext.
   * @param {Logger} [config.logger] - Optional logger for logging purposes.
   * @param {string} [config.cacheDir] - Optional directory path for caching.
   * @param {VectorStore} [config.vectorStore] - Optional vector store instance.
   * @param {string} [config.summarizationPrompt] - Optional prompt for summarization.
   * @param {Model} [config.summaryModel] - Optional model for generating summaries.
   * @param {Model} [config.embeddingsModel] - Optional model for generating embeddings.
   *
   * @returns {Promise<ReadNext>} A promise that resolves to an instance of ReadNext.
   */
  static async create(config: CreateReadNextArgs = {}): Promise<ReadNext> {
    let { logger, cacheDir, vectorStore, summarizationPrompt, summaryModel, parallel, getSourceDocument } = config;

    if (!config.embeddingsModel) {
      config.embeddingsModel = new OpenAIEmbeddings({ model: "text-embedding-ada-002" });
    }

    if (!vectorStore && cacheDir) {
      const faissPath = path.join(cacheDir, "faiss.index");

      if (fs.existsSync(faissPath)) {
        vectorStore = await FaissStore.load(cacheDir, config.embeddingsModel);
      }
    }

    if (!vectorStore) {
      vectorStore = new FaissStore(config.embeddingsModel, {});
    }

    if (!summaryModel) {
      summaryModel = new ChatOpenAI({ model: "gpt-4o", temperature: 0.7 });
    }

    const readNextConfig: ReadNextArgs = {
      vectorStore,
      summaryModel,
      cacheDir,
      summarizationPrompt,
      logger,
      parallel,
      getSourceDocument,
    };

    return new ReadNext(readNextConfig);
  }

  /**
   * Constructs an instance of the ReadNext class.
   *
   * @param vectorStore - The vector store used for embeddings.
   * @param summaryModel - The model used for generating summaries.
   * @param summarizationPrompt - The prompt used for summarization, defaults to `defaultSummarizationPrompt`.
   * @param cacheDir - The directory used for caching, defaults to the system's temporary directory.
   * @param logger - The logger instance, defaults to a Winston logger with console transport.
   */
  constructor({
    vectorStore,
    summaryModel,
    summarizationPrompt = defaultSummarizationPrompt,
    cacheDir,
    logger,
    parallel = 1,
    getSourceDocument,
    sourceDocuments,
  }: ReadNextArgs) {
    this.vectorStore = vectorStore;
    this.summaryModel = summaryModel;
    this.summarizationPrompt = summarizationPrompt;
    this.embeddingsModel = this.vectorStore.embeddings;

    this.summaryParser = new StringOutputParser();
    this.summaryChain = this.summaryModel.pipe(this.summaryParser);

    this.cacheDir = cacheDir || path.join(os.tmpdir(), "read-next-cache");

    fs.mkdirSync(this.cacheDir, { recursive: true });

    this.logger = logger || readNextLogger;
    this.parallel = parallel;

    if (typeof getSourceDocument === "function") {
      this.getSourceDocument = getSourceDocument; // || ((doc: any) => doc);
    }
    this.sourceDocuments = sourceDocuments || [];

    this.contentHasher = new ContentHasher({ cacheDir: this.cacheDir, logger: this.logger });
  }

  /**
   * Retrieves the summary content by its identifier.
   *
   * @param id - The unique identifier of the summary.
   * @returns The content of the summary as a string if it exists, otherwise undefined.
   */
  getSummaryById(id: string) {
    const summaryFileName = path.join(this.cacheDir, "summaries", id);

    if (fs.existsSync(summaryFileName)) {
      return fs.readFileSync(summaryFileName, "utf8");
    }
  }

  /**
   * Indexes the provided source documents by generating summaries, adding them to a vector store,
   * and saving the state to a cache. Supports parallel execution.
   *
   * @param {Object} params - The parameters for the index function.
   * @param {DocumentInput[]} params.sourceDocuments - An array of source documents to be indexed.
   * @param {number} [params.parallel=1] - The number of documents to process in parallel. Uses a pooling mechanism.
   * @param {string|Function} [params.summarizationPrompt] - Optional override to default summarizationPrompt
   *
   * @returns {Promise<Document[]>} A promise that resolves to an array of summary documents.
   */
  async index({
    sourceDocuments = this.sourceDocuments,
    parallel,
    summarizationPrompt,
  }: {
    sourceDocuments?: DocumentInput[];
    parallel?: number;
    summarizationPrompt?: string | ((doc: DocumentInput) => string);
  }) {
    const summaryDocuments: Document[] = [];
    let embeddingsAdded = 0;

    // Indexes a single document
    const processDocument = async (doc: DocumentInput) => {
      const sourceDocument = this.getSourceDocument(doc);
      const summary = await this.getSummaryFor({ sourceDocument: doc, summarizationPrompt });

      const summaryDocument: Document = {
        pageContent: summary,
        metadata: {
          sourceDocumentId: sourceDocument.id,
        },
      };

      this.logger.info(`Adding ${sourceDocument.id} to the vector store`, { id: sourceDocument.id });
      try {
        await this.vectorStore.delete({ ids: [sourceDocument.id] });
      } catch (e) {
        // ignore errors if the document doesn't already exist
      }
      await this.vectorStore.addDocuments([summaryDocument], { ids: [sourceDocument.id] });
      this.logger.info(`${sourceDocument.id} added to the vector store`, { id: sourceDocument.id });

      // Save to cache
      this.contentHasher.set(sourceDocument);
      this.contentHasher.save();

      return summaryDocument;
    };

    // Process documents in parallel, using a task pool
    let index = 0; // Track the index of the current document

    const processNext = async () => {
      if (index >= sourceDocuments.length) {
        return;
      }

      const currentDocument = sourceDocuments[index];
      index++;
      const summaryDocument = await processDocument(currentDocument);

      summaryDocuments.push(summaryDocument);
      embeddingsAdded += 1;

      await processNext();
    };

    // Initialize the task pool with 'parallel' tasks
    const tasks = Array(Math.min(parallel || this.parallel, sourceDocuments.length))
      .fill(null)
      .map(() => processNext());

    // Wait for all tasks to complete
    await Promise.all(tasks);

    if (embeddingsAdded > 0) {
      // Save vector store if new embeddings were added
      try {
        // @ts-ignore
        if (typeof this.vectorStore.save === "function") {
          // @ts-ignore
          await this.vectorStore.save(this.cacheDir);
        }
      } catch (e) {
        this.logger.error("Error saving vector store", { error: e });
      }
    }

    return summaryDocuments;
  }

  /**
   * Generates a summary for the given source document. If the document has an ID and a fresh hash,
   * it attempts to retrieve a cached summary from the filesystem. If no cached summary is found,
   * it generates a new summary and caches it if the document has an ID. The content hash is updated
   * and saved if it is not fresh.
   *
   * @param {Object} options - The input object containing the source document.
   * @param {DocumentInput} options.sourceDocument - The document for which to generate a summary.
   * @returns {Promise<string>} - A promise that resolves to the summary of the document.
   */
  async getSummaryFor({ sourceDocument, summarizationPrompt }: GetSummaryArgs): Promise<string> {
    let summary: string | undefined = undefined;
    sourceDocument = this.getSourceDocument(sourceDocument);

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
      this.logger.warn("No id found for document, summary will not be cached");
    }

    if (summary === undefined) {
      this.logger.info(`No summary found for ${sourceDocument.id}, will generate a new one`, {
        cache: "miss",
        id: sourceDocument.id,
      });
      summary = await this.summarize({ sourceDocument, summarizationPrompt });
    } else {
      this.logger.info(`Using cached summary for ${sourceDocument.id}`, { cache: "hit", id: sourceDocument.id });
    }

    if (hasId) {
      fs.mkdirSync(path.dirname(summaryFileName), { recursive: true });
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
   * Summarizes a source document, saving it to the cacheDir.
   * @returns the summary of the source document
   */
  async summarize({ sourceDocument, summarizationPrompt = this.summarizationPrompt }: Summarize): Promise<string> {
    const { pageContent } = sourceDocument;

    const messages = [
      new SystemMessage(
        typeof summarizationPrompt === "string" ? summarizationPrompt : summarizationPrompt(sourceDocument)
      ),
      new HumanMessage(pageContent),
    ];

    this.logger.info(`Generating summary for ${sourceDocument.id}`, { expensive: true, id: sourceDocument.id });
    const summary = await this.summaryChain.invoke(messages);
    this.logger.info("Summarization completed", { id: sourceDocument.id });

    return summary;
  }

  async prompt({ sourceDocuments = this.sourceDocuments, prompt, docId }: PromptArgs) {
    //make sure the source documents are all indexed
    await this.index({ sourceDocuments });

    //get the summaries for these posts
    const summaries = await Promise.all(
      sourceDocuments.map(async (sourceDocument) => this.getSummaryFor({ sourceDocument }))
    );

    const message =
      typeof prompt === "string" ? prompt : prompt(sourceDocuments?.map(this.getSourceDocument), summaries);

    this.logger.info("invoking prompt");
    const summary = await this.summarize({
      sourceDocument: { pageContent: message, id: docId },
      summarizationPrompt: message,
    });

    this.logger.info("prompt response received");

    return summary;
  }

  /**
   * Suggests related documents based on the provided source document.
   *
   * @param {Object} params - The parameters for the suggestion.
   * @param {Object} params.sourceDocument - The source document to base suggestions on.
   * @param {number} [params.limit=1] - The maximum number of suggestions to return.
   * @returns {Promise<Suggestions>} A promise that resolves to an object containing the source document ID and an array of related document suggestions with their scores.
   */
  async suggest({ sourceDocument, limit = 1 }: Suggest): Promise<Suggestions> {
    this.logger.info(`Getting suggestion for ${sourceDocument.id}`, { id: sourceDocument.id });
    const summary = await this.getSummaryFor({ sourceDocument });
    const results = await this.vectorStore.similaritySearchWithScore(summary, limit + 1);

    const related = results
      .filter(([doc]) => doc.metadata.sourceDocumentId !== sourceDocument.id)
      .map(([{ metadata }, score]) => ({ sourceDocumentId: metadata.sourceDocumentId, score, metadata }));

    return {
      id: sourceDocument.id,
      related: related.slice(0, limit + 1),
    };
  }
}

/**
 * Represents a document that is related to another document.
 *
 * @typedef {Object} RelatedDocument
 * @property {string} sourceDocumentId - The unique identifier of the source document.
 * @property {number} score - The relevance score of the related document.
 */
type RelatedDocument = {
  sourceDocumentId: string;
  score: number;
  metadata: any;
};

/**
 * Represents a collection of suggestions.
 *
 * @typedef {Object} Suggestions
 * @property {string} [id] - Optional identifier for the suggestion.
 * @property {RelatedDocument[]} related - Array of related documents.
 */
type Suggestions = {
  id?: string;
  related: RelatedDocument[];
};
