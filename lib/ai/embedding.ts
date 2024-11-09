import { openai } from "@ai-sdk/openai"
import { embed, embedMany } from "ai"
import { cosineDistance, desc, gt, sql } from "drizzle-orm"
import { embeddings } from "../db/schema/embeddings"
import { db } from "../db"

const generateChunks = (input: string): string[] => {
  return input
    .trim()
    .split(".")
    .filter(i => i !== "")
}

const embeddingModel = openai.embedding("text-embedding-ada-002")

export const generateEmbeddings = async (
  value: string
) => {
  const chunks = generateChunks(value)
  const {
    embeddings
  } = await embedMany({
    model: embeddingModel,
    values: chunks
  })
  return embeddings.map((e, i) => ({ content: chunks[i], embedding: e }))
}

export const generateEmbedding = async (value: string) => {
  const input = value.replaceAll("\\n", "")
  const { embedding } = await embed({
    model: embeddingModel,
    value: input
  })
  return embedding
}

export const findRelevantContent = async (userQuery: string) => {
  const userQueryEmbedded = await generateEmbedding(userQuery)
  const similarity = sql<number>`1 - (${cosineDistance(
    embeddings.embedding,
    userQueryEmbedded
  )})`
  console.log({ similarity })
  const similarGuides = await db
    .select({ name: embeddings.content, similarity })
    .from(embeddings)
    .where(gt(similarity, 0.5))
    .orderBy(t => desc(t.similarity))
    .limit(4)
  return similarGuides
}