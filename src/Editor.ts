import fs from "fs";
import path from "path";
import YAML from "yaml";

import readline from "readline";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

interface EditorArgs {
  cacheDir: string;
  parallel?: number;
  outputDir?: string;
  examplesDir?: string;
  exampleTemplate?: Function;
  temperature?: number;
  model?: any;
}

export type Example = {
  docId: string;
  content: string;
  reason?: string;
  verdict: string;
};

export type LLMTask = {
  docId: string;
  prompt: string;
  suffix?: string;
};

import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";

const defaultTemperature = 0.9;
const defaultExampleTemplate = (example: Example) => `
<example>
  <content>${example.content}</content>
  <reason>${example.reason || "No reason given"}</reason>
</example>`;

const examplesForKey = (examplesDir: string, key: string) => path.join(examplesDir, key + ".yml");
const examplesByVerdict = (examplesDir: string, key: string, verdict: string) =>
  examplesForKey(examplesDir, verdict + "/" + key);

async function getExampleVerdict(): Promise<any> {
  return new Promise((resolve) => {
    rl.question("Save as [g]ood example, save as [b]ad example, or [s]kip (g/b/s): ", async (answer) => {
      let verdict;

      if (answer === "g" || answer === "b") {
        verdict = answer === "g" ? "good" : "bad";

        const reason = await getVerdictReason();

        return resolve({ verdict, reason: reason === "" ? undefined : reason, choice: "save" });
      }

      return resolve({ choice: "skip" });
    });
  });
}

async function getVerdictReason(): Promise<string> {
  return new Promise((resolve, reject) => {
    rl.question("Reasoning (optional, Enter to skip): ", (reason) => {
      resolve(reason);
    });
  });
}

export class EditorAI {
  cacheDir: string;
  parallel: number = 1;
  outputDir: string | undefined;
  examplesDir: string | undefined;
  exampleTemplate: Function;
  temperature: any;
  model: any;

  constructor({
    cacheDir,
    parallel,
    outputDir,
    examplesDir,
    exampleTemplate = defaultExampleTemplate,
    temperature = defaultTemperature,
    model,
  }: EditorArgs) {
    this.cacheDir = cacheDir;
    this.parallel = parallel || this.parallel;

    this.outputDir = outputDir;
    this.examplesDir = examplesDir;

    this.exampleTemplate = exampleTemplate || defaultExampleTemplate;

    this.temperature = temperature;
    this.model = model || openai("gpt-4o");
  }

  async generateMulti(tasks: LLMTask[]) {
    for (const task of tasks) {
    }
  }

  async train(task: LLMTask) {
    const { docId } = task;

    console.log(`Generating content for ${docId}...\n`);

    const content = await this.generate(task);

    console.log(`Proposed content for ${docId}:\n`);
    console.log(content);
    console.log("\n\n");

    const answer = await getExampleVerdict();

    if (answer.choice === "save") {
      const { verdict, reason } = answer;
      const saveOutcome = await this.saveExample({ docId, verdict, content, reason });

      console.log(saveOutcome ? "Saved example" : "Failed to save example");
    }
  }

  async generate(task: LLMTask, save = false) {
    const { prompt, suffix } = task;

    console.log("generating");
    console.log(task.docId);
    // console.log(task.prompt);

    const promptElements = [prompt];

    const goodExamples = await this.getExamplesForTask(task, "good");
    const badExamples = await this.getExamplesForTask(task, "bad");

    if (goodExamples.length) {
      promptElements.push("Here are a few examples of what your response should look like:");
      promptElements.push(goodExamples.map((example) => this.exampleTemplate(example)).join("\n"));
    }

    if (badExamples.length) {
      promptElements.push("Here are a few examples of what your response should not look like:");
      promptElements.push(badExamples.map((example) => this.exampleTemplate(example)).join("\n"));
    }

    if (suffix) {
      promptElements.push(suffix);
    }

    const { model, temperature } = this;

    const { text } = await generateText({
      model,
      temperature,
      prompt: promptElements.join("\n\n"),
    });

    return text;
  }

  async saveExample({
    docId,
    content,
    verdict,
    reason,
  }: {
    docId: string;
    content: string;
    verdict: string;
    reason: string;
  }) {
    const exampleKey = this.getExampleGroup(docId);
    const verdictDir = verdict === "good" ? "good" : "bad";

    if (!this.examplesDir || !exampleKey) {
      return false;
    }

    //save the content
    const exampleSaveDir = path.join(this.examplesDir, verdictDir);
    const examplesSavePath = examplesForKey(exampleSaveDir, exampleKey);

    //read the existing yaml
    let existingExamples = this.readExampleYaml(examplesSavePath);

    existingExamples.push({ docId, content, reason });

    try {
      fs.mkdirSync(exampleSaveDir, { recursive: true });
      fs.writeFileSync(examplesSavePath, YAML.stringify(existingExamples));
    } catch (e) {
      console.error("error saving example", e);
      return false;
    }

    return true;
  }

  readExampleYaml(filename: string) {
    try {
      return YAML.parse(fs.readFileSync(filename, "utf-8"));
    } catch (e) {
      return [];
    }
  }

  async getExamplesForTask(task: LLMTask, verdict: string): Promise<Example[]> {
    if (!this.examplesDir) {
      return [];
    }

    const examplesKey = this.getExampleGroup(task.docId);

    if (examplesKey) {
      const examples = this.readExampleYaml(examplesByVerdict(this.examplesDir, examplesKey, verdict));

      return examples.map((example: Example) => ({ ...example, verdict: "good" }));
    } else {
      return [];
    }
  }

  getExampleGroup(docId?: string) {
    return docId?.split("/")[0];
  }

  getExampleKey(docId: string) {
    const splits = docId.split("/");

    return splits[splits.length - 1];
  }
}
