import { SearchResponse, BenchmarkResponse } from "./types";

const HEADERS = {
  "Content-Type": "application/json",
};

export const checkHealth = async (baseUrl: string): Promise<boolean> => {
  try {
    const response = await fetch(`${baseUrl}/health`, {
      method: "GET",
      headers: HEADERS,
    });
    const data = await response.json();
    return data.status === "ok";
  } catch (error) {
    console.error("Health check failed", error);
    return false;
  }
};

export const searchArticles = async (
  baseUrl: string,
  query: string,
  k: number = 5
): Promise<SearchResponse> => {
  const response = await fetch(`${baseUrl}/search`, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({ query_text: query, k_neighbors: k }),
  });
  if (!response.ok) throw new Error("Search request failed");
  return response.json();
};

export const runBenchmark = async (
  baseUrl: string,
  numQueries: number = 100
): Promise<BenchmarkResponse> => {
  const response = await fetch(`${baseUrl}/benchmark`, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({ num_queries: numQueries }),
  });
  if (!response.ok) throw new Error("Benchmark request failed");
  return response.json();
};