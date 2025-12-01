export interface SearchResultItem {
  rank: number;
  text: string;
  similarity: number;
  dataset_index: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
  search_time_ms: number;
}

export interface BenchmarkResultItem {
  method: string;
  time_seconds: number;
  avg_ms_per_query: number;
  recall: number;
  memory_mb: number;
}

export interface BenchmarkResponse {
  dataset_size: number;
  num_queries: number;
  compression_ratio: number;
  results: BenchmarkResultItem[];
}