# ReadNext - AI-driven "Read Next" for your content

**ReadNext** is a tool that uses AI to create "Read Next" suggestions for your articles. It gives you a simple way to index all your content and automatically generate recommendations for what your audience should read next based on what they're reading right now.

### Key Features:

- **Minimal configuration**: Just Works OOTB, though a `cacheDir` is recommended
- **Caches results and recommendations**: Saves your time and money by not repeating work
- **Use an LLM of your choice**: for both summarization and embedding (sensible defaults provided)
- **Built on top of [Langchain](https://js.langchain.com/v0.2/docs/introduction)**: mature foundation with plenty of room for customization

It's not a UI component itself, but powers "What to Read Next" UI components like this:

![Example UI using Read Next](/docs/read-next-ui-example.png)

## Installation

Install the NPM package:

```sh
npm install read-next
```

If you want to use customized summary and/or embedding models, you'll likely need to install the appropriate langchain packages too (see configuration examples below).

## Usage

Generating recommendations using ReadNext is straightforward:

```tsx
import { ReadNext } from "read-next";

async function generateRecommendations() {
  //create the ReadNext instance
  const readNext = await readNext.create({
    cacheDir: path.join(__dirname, "read-next"), //optional, but recommended
  });

  //index your content - whenever you add/change articles or other content
  await readNext.index({
    sourceDocuments: [
      {
        id: "a-wonderful-article",
        pageContent: "The article content is here...",
      },
      {
        id: "my-latest-article",
        pageContent: "Article content goes here - can be long, it will be summarized first",
      },
      // ...as many documents as you like here
    ],
  });

  //get suggestions for an article
  const suggestions = await readNext.suggest({
    sourceDocument: {
      id: "my-latest-article",
    },
    limit: 3,
  });
}
```

You'll get back an object like this, with related articles ranked by score (lower is better):

```json
{
  "id": "my-latest-article",
  "related": [
    {
      "sourceDocumentId": "a-wonderful-article",
      "score": 0.590001106262207
    },
    {
      "sourceDocumentId": "a-less-related-article",
      "score": 0.7498645782470703
    },
    {
      "sourceDocumentId": "a-really-unrelated-article",
      "score": 1.3464351892471313
    }
  ]
}
```

Because articles don't change very often and processing them can take some time, ReadNext keeps a cache of everything it does so that it doesn't have to repeat work. If you don't supply a `cacheDir` it will just dump that into the system tmpdir, but you're better off supplying a directory of your own.

## How it works

ReadNext builds on top of LangChain, and uses [FAISS](https://github.com/facebookresearch/faiss) as a local vector store. When you index your content with it, ReadNext does the following:

1. Creates a summary of your article using a `summaryModel` of your choice (defaults to OpenAI's `gpt-4o-mini`)
2. Saves the summary and the sha256 hash of the article content that spawned it, so we don't have to summarize again unless the article content changes
3. Adds the summary to a local FAISS vector store, using an embedding model of your choice (defaults to OpenAI's `text-embedding-ada-002`)

After the content has been indexed, the `suggest` function takes an article and returns as many similar articles as you ask for, based on the vector similarity between the summaries it generated for your content.

ReadNext performs all of this locally (aside from the calls to the LLMs), so you don't need any vector database infrastructure to exist. This means it's well-suited to being run on your laptop, in a CI job, or pretty much anywhere else.

## Configuration

### Parallel

If you are indexing lots of documents, it will be a lot faster if you run them in parallel. You can configure this when you create ReadNext:

```tsx
const readNext = await ReadNext.create({
  cacheDir: "/my/cache/dir",
  parallel: 10,
});

await readNext.index({ sourceDocuments: myHugeArrayOfDocuments });
```

This will kick off 10 indexing jobs at the same time, starting another one each time one finishes. This will result in 10 calls to your summaryModel LLM happening simultaneously, so be aware of rate limits here.

You can also override it when you call `index`:

```tsx
await readNext.index({
  sourceDocuments: myHugeArrayOfDocuments,
  parallel: 5,
});
```

Defaults to 1 (e.g. not parallel). It does make the grouped logging a little less pretty, but you can't win them all. It's a lot faster.

### Summary Model

If you don't supply a model to perform summarization with, ReadNext will default to using OpenAI's `gpt-4o-mini`, because it's relatively fast and cheap. You will need to make sure that you have an `OPENAI_API_KEY` environment variable available for Langchain to use, otherwise it will give an error.

If you want to use a different model, just pass it in like this:

```tsx
import { ChatAnthropic } from "@langchain/anthropic";

const readNext = await ReadNext.create({
  cacheDir: path.join(__dirname, "read-next"),

  summaryModel: new ChatAnthropic({
    model: "claude-3-haiku-20240307",
    temperature: 0,
    maxTokens: undefined,
    maxRetries: 2,
    // other params...
  }),
});
```

Make sure you set up whatever environment variables your model of choice expects to be present (`ANTHROPIC_API_KEY` in the case above).

### Embeddings Model

If you don't want to use the default `text-embedding-ada-002` OpenAI model for embeddings, you can specify your own like this:

```tsx
import { VertexAIEmbeddings } from "@langchain/google-vertexai";

const readNext = await ReadNext.create({
  cacheDir: path.join(__dirname, "read-next"),

  embeddingsModel: new VertexAIEmbeddings({
    model: "text-embedding-004",
    // other params...
  }),
});
```

Make sure you set up whatever environment variables your model of choice expects to be present (`GOOGLE_APPLICATION_CREDENTIALS` in the case above).

### Summarization Prompt

ReadNext has a reasonable default summarization prompt that it sends to the summaryModel LLM to summarize your content, but you can often get a better outcome by using something more specific to your use case.

For example, the [RSC Examples project](https://github.com/edspencer/rsc-examples) is a collection of React Server Component examples - basically a bunch of .mdx files with associated code snippets. The examples are quite similar to articles but it's helpful to give the LLM a little more specific context of what it is summarizing. Here's how RSC Examples does that ([see the actual code here](https://github.com/edspencer/rsc-examples/blob/main/src/script/related.ts)).

```tsx
const readNext = await ReadNext.create({
  cacheDir: path.join(__dirname, "read-next"),

  //this will be sent to the LLM just before your sourceDocument's pageContent
  summarizationPrompt: `
    The following content is a markdown document about an example of how to use React Server
    Components. It contains sections of prose explaining what the example is about, may contain
    links to other resources, and almost certainly contains code snippets.

    Your goal is to generate a summary of the content that can be used to suggest related examples.
    The summary will be used to create embeddings for a vector search. When you come across code
    samples, please summarize the code in natural language.

    Do not reply with anything except your summary of the example.`,
});
```

### Cache directory

If you don't supply a `cacheDir` argument, ReadNext will save its temporary files into the system tmpdir. Invoking LLMs and vector databases can be expensive in both time and money though, so it's always a good idea to supply this:

```tsx
const readNext = await ReadNext.create({
  cacheDir: path.join(__dirname, "some-dir"),
});
```

It's recommended use a `cacheDir` inside your source-controlled directory so that ReadNext can easily skip work it doesn't need to perform again.

### Custom logger

ReadNext uses winston for logging, and by default will just log to the console, but you can pass it any winston logger object like so (for example if you want to save log files):

```tsx
const cacheDir = path.join(__dirname, "read-next");

const readNext = await ReadNext.create({
  cacheDir,
  logger: winston.createLogger({
    level: "info",
    transports: [
      new winston.transports.File({
        format: winston.format.simple(),
        filename: path.join(cacheDir, "readnext-error.log"),
        level: "error",
      }),
      new winston.transports.File({
        format: winston.format.simple(),
        filename: path.join(cacheDir, "readnext.log"),
      }),
      new winston.transports.Console({ format: winston.format.cli() }),
    ],
  }),
});
```

### Vector Store

If you don't want to use FAISS for the vector store, you can swap it out for another subclass of [VectorStore](https://js.langchain.com/v0.2/docs/integrations/vectorstores/). This isn't likely to be something most people want/need to do, but here's how you would do it:

```tsx
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";

const readNext = await ReadNext.create({
  cacheDir: path.join(__dirname, "some-dir"),

  vectorStore: new MemoryVectorStore(
    new OpenAIEmbeddings({
      model: "text-embedding-3-small",
    })
  ),
});
```

Compatibility is not guaranteed for all VectorStores, though SaveableVectorStore subclasses should be good to go.

## Tips and tricks

### Source control cacheDir

The `cacheDir` will be populated with a few files:

- 2 files for FAISS to persist its vector index
- 1 file for ReadNext to keep the latest sha hashes of your content
- N files for saved summarization outputs (inside the `summaries` subdirectory)

It's recommended to check the entire ReadNext directory into source control so that it's easy to rebuild recommendations from wherever your repo is checked out. Otherwise, ReadNext may have to re-summarize and re-index everything, which could be slow and potentially costly.

### Use a script to rebuild recommendations

Whenever you add or update content, it makes sense to run ReadNext again as the recommendations for any given article may have changed.

In your `package.json` file, add something like this:

```json
"scripts": {
  "readnext": "npx tsx script/generate-recommendations.ts"
}
```

Now you can regenerate recommendations easily on the command line or in your CI process:

```sh
npm run readnext
```

The script itself might look like (this is the actual script used to create the recommendations on https://edspencer.net):

```ts
import * as dotenv from "dotenv";
import { ReadNext } from "read-next";
import path from "path";
import Posts from "@/lib/blog/Posts";

dotenv.config();

const cacheDir = path.join(__dirname, "..", "read-next");

async function main() {
  const posts = new Posts();
  const { publishedPosts } = posts;

  let sourceDocuments: any[] = [];

  const readNext = await ReadNext.create({
    cacheDir,
  });

  //map our content into the format that ReadNext expects
  sourceDocuments = publishedPosts.map((post: any) => ({
    pageContent: posts.getContent(post),
    id: post.slug,
  }));

  //index all the content
  await readNext.index({ sourceDocuments });

  for (const post of publishedPosts) {
    //get the top 5 recommendations for each published post
    const suggestions = await readNext.suggest({
      sourceDocument: sourceDocuments.find((s: any) => s.id === post.slug),
      limit: 5,
    });

    //update the frontmatter on the source .mdx file
    await posts.updateMatter(post, {
      related: suggestions.related.map((suggestion: any) => suggestion.sourceDocumentId),
    });
  }
}

main()
  .catch(console.error)
  .then(() => process.exit(0));
```

### Save the recommendations with your content

The output of `suggest` looks like this (actual output taken from the content for https://edspencer.net):

```json
{
  "id": "rails-asset-tag-expansions",
  "related": [
    {
      "sourceDocumentId": "useful-rails-javascript-expansions-for",
      "score": 0.590001106262207
    },
    {
      "sourceDocumentId": "sencha-con-2013-ext-js-performance-tips",
      "score": 0.7498645782470703
    },
    {
      "sourceDocumentId": "drying-up-your-crud-controller-rspecs",
      "score": 0.8464351892471313
    },
    {
      "sourceDocumentId": "autotesting-javascript-with-jasmine-and-guard",
      "score": 0.8512178063392639
    },
    {
      "sourceDocumentId": "writing-compressible-javascript",
      "score": 0.8858739733695984
    }
  ]
}
```

The scores may be useful in deciding whether or not to keep all of the recommendations. The lower the score the better. Your articles probably already have metadata like tags, publication status and other things - if so that's a good place to store the recommendations too. Check out [this blog post](https://edspencer.net/2024/8/28/using-markdown-with-nextjs) on how I manage metadata for my blog with MDX and frontmatter.

## Troubleshooting

If you come across this error:

`` Error: Vector store not initialised yet. Try calling `fromTexts`, `fromDocuments` or `fromIndex` first. ``

You are probably calling `suggest` before you have `index`ed anything. Make sure you have actually indexed some content first.
