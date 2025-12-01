import React, { useState } from 'react';
import { checkHealth } from '../services/api';

interface ConnectBackendProps {
  onConnect: (url: string) => void;
}

const COLAB_SCRIPT = `# -*- coding: utf-8 -*-
"""ScaNN Server for Colab - Assignment Submission"""

# --- 1. INSTALL DEPENDENCIES ---
# !pip install fastapi uvicorn pyngrok scann sentence-transformers datasets nest_asyncio

import os
import shutil
import time
import numpy as np
import uvicorn
import nest_asyncio
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import scann
from datasets import load_dataset
from pyngrok import ngrok
import torch
from contextlib import asynccontextmanager

# Apply nest_asyncio immediately to allow nested event loops in Colab
nest_asyncio.apply()

# --- 2. CONFIGURATION ---
# Sign up at ngrok.com to get your token (optional but recommended)
NGROK_AUTH_TOKEN = "" # @param {type:"string"}
DATASET_SIZE = 20000  # @param {type:"integer"}
MODEL_NAME = 'all-MiniLM-L6-v2'
K_NEIGHBORS = 10 

# Global State
embedding_model = None
searcher = None
dataset_texts = []
normalized_dataset_embeddings = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, searcher, dataset_texts, normalized_dataset_embeddings
    print("--- SERVER STARTUP ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"1. Init Model ({device})...")
    embedding_model = SentenceTransformer(MODEL_NAME, device=device)
    
    print(f"2. Load Data ({DATASET_SIZE} items from ag_news)...")
    try:
        # Load the dataset
        dataset = load_dataset('ag_news', split=f'train[:{DATASET_SIZE}]')
        dataset_texts = dataset['text']
        
        print("3. Embed & Normalize (Pre-processing)...")
        emb = embedding_model.encode(dataset_texts, convert_to_tensor=False, show_progress_bar=True)
        # L2 Normalization for Dot Product to simulate Cosine Similarity
        normalized_dataset_embeddings = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        
        print("4. Build ScaNN Index (Partitioning -> Scoring -> Reordering)...")
        # Theory: Partitioning (Tree)
        num_leaves = int(np.sqrt(DATASET_SIZE)) 
        
        builder = scann.scann_ops_pybind.builder(
            normalized_dataset_embeddings, K_NEIGHBORS, "dot_product"
        ).tree(
            num_leaves=num_leaves, 
            num_leaves_to_search=max(int(num_leaves*0.1), 10), 
            training_sample_size=min(int(DATASET_SIZE*0.1), 10000)
        ).score_ah(
            2, anisotropic_quantization_threshold=0.2 # Theory: Anisotropic Hashing
        ).reorder(
            K_NEIGHBORS * 5 # Theory: Reordering
        )
        
        searcher = builder.build()
        print("--- READY ---")
    except Exception as e:
        print(f"Error during startup: {e}")
    
    yield
    print("--- SERVER SHUTDOWN ---")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query_text: str
    k_neighbors: int = 5

class BenchmarkRequest(BaseModel):
    num_queries: int = 100

@app.get("/health")
def health_check():
    return {"status": "ok", "ready": searcher is not None}

@app.post("/search")
def search(query: SearchQuery):
    if not searcher: raise HTTPException(503, "Not ready. Model might still be loading.")
    start = time.time()
    
    # 1. Embed Query
    q_emb = embedding_model.encode([query.query_text])[0]
    q_emb = q_emb / np.linalg.norm(q_emb)
    
    # 2. Search
    indices, distances = searcher.search(q_emb, final_num_neighbors=min(query.k_neighbors, K_NEIGHBORS))
    
    elapsed = (time.time()-start)*1000
    
    # Cast idx to int to avoid numpy.uint32 TypeError
    results = [{"rank": i+1, "text": dataset_texts[int(idx)], "similarity": float(d), "dataset_index": int(idx)} 
               for i, (idx, d) in enumerate(zip(indices, distances))]
               
    return {"query": query.query_text, "results": results, "search_time_ms": elapsed}

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    if not searcher: raise HTTPException(503, "Not ready")
    print(f"--- STARTING BENCHMARK ({req.num_queries} queries) ---")
    
    try:
        # Load test set
        test_texts = load_dataset('ag_news', split=f'test[:{req.num_queries}]')['text']
        test_emb = embedding_model.encode(test_texts, convert_to_tensor=False)
        test_emb = test_emb / np.linalg.norm(test_emb, axis=1, keepdims=True)
        
        # 1. Brute Force (Ground Truth)
        print("Running Brute Force...")
        bf_start = time.time()
        # Create a raw brute force searcher for comparison
        bf_searcher = scann.scann_ops_pybind.builder(normalized_dataset_embeddings, K_NEIGHBORS, "dot_product").score_brute_force().build()
        bf_idx, _ = bf_searcher.search_batched(test_emb)
        bf_time = time.time() - bf_start
        
        # 2. ScaNN (Approximate)
        print("Running ScaNN...")
        scann_start = time.time()
        scann_idx, _ = searcher.search_batched(test_emb)
        scann_time = time.time() - scann_start
        
        # 3. Calculate Recall
        recall_sum = 0
        k = 5
        for i in range(len(test_texts)):
            # Intersection of indices
            recall_sum += len(set(bf_idx[i][:k]).intersection(set(scann_idx[i][:k]))) / k
            
        avg_recall = recall_sum/len(test_texts)
        
        # 4. Memory & Compression Stats
        print("Calculating Memory Stats...")
        # Raw Embeddings Size (float32 = 4 bytes)
        raw_size_bytes = normalized_dataset_embeddings.nbytes
        raw_size_mb = raw_size_bytes / (1024 * 1024)
        
        # ScaNN Index Size (Serialize to temp to measure)
        scann_dir = "/tmp/scann_temp_index"
        if os.path.exists(scann_dir): shutil.rmtree(scann_dir)
        searcher.serialize(scann_dir)
        
        scann_size_bytes = 0
        for dirpath, _, filenames in os.walk(scann_dir):
            for f in filenames:
                scann_size_bytes += os.path.getsize(os.path.join(dirpath, f))
        
        scann_size_mb = scann_size_bytes / (1024 * 1024)
        compression = raw_size_bytes / scann_size_bytes if scann_size_bytes > 0 else 0
        
        print(f"Benchmark Complete. Recall: {avg_recall:.4f}, Compression: {compression:.2f}x")
        
        return {
            "dataset_size": DATASET_SIZE, 
            "num_queries": req.num_queries,
            "compression_ratio": compression,
            "results": [
                {
                    "method": "Brute Force", 
                    "time_seconds": bf_time, 
                    "avg_ms_per_query": (bf_time / req.num_queries) * 1000,
                    "recall": 1.0,
                    "memory_mb": raw_size_mb
                },
                {
                    "method": "ScaNN", 
                    "time_seconds": scann_time, 
                    "avg_ms_per_query": (scann_time / req.num_queries) * 1000,
                    "recall": avg_recall,
                    "memory_mb": scann_size_mb
                }
            ]
        }
    except Exception as e:
        print(f"Benchmark Error: {e}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    if NGROK_AUTH_TOKEN: ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    # Connect ngrok
    public_url = ngrok.connect(8000).public_url
    print(f"URL: {public_url}")
    
    # Run Uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()
`;

export const ConnectBackend: React.FC<ConnectBackendProps> = ({ onConnect }) => {
  const [url, setUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [copySuccess, setCopySuccess] = useState(false);

  const handleConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    const cleanUrl = url.replace(/\/$/, "");

    try {
      const isHealthy = await checkHealth(cleanUrl);
      if (isHealthy) {
        onConnect(cleanUrl);
      } else {
        setError('Backend reachable, but returned unhealthy status. Models might still be loading.');
      }
    } catch (err) {
      console.error(err);
      setError('Could not connect. Ensure the URL is correct, the Colab cell is running, and you are not blocked by a firewall.');
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(COLAB_SCRIPT);
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2000);
  };

  return (
    <div className="max-w-4xl mx-auto mt-10 grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Left: Connection Form */}
      <div className="p-6 bg-slate-800 rounded-xl shadow-lg border border-slate-700 h-fit">
        <h2 className="text-2xl font-bold mb-4 text-white">1. Connect</h2>
        <p className="text-slate-400 mb-6 text-sm">
          Paste the public URL (ngrok) generated by the Python script below.
        </p>
        
        <form onSubmit={handleConnect} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">API URL</label>
            <input
              type="text"
              placeholder="https://xxxx-xx-xx-xx-xx.ngrok-free.app"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-all"
              required
            />
          </div>
          
          {error && (
            <div className="p-3 bg-red-900/50 border border-red-500/50 rounded-lg text-red-200 text-sm">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-2 px-4 rounded-lg font-semibold text-white transition-all
              ${isLoading 
                ? 'bg-indigo-500/50 cursor-not-allowed' 
                : 'bg-indigo-600 hover:bg-indigo-500 shadow-lg shadow-indigo-500/30'
              }`}
          >
            {isLoading ? 'Connecting...' : 'Connect'}
          </button>
        </form>
      </div>

      {/* Right: Code Snippet */}
      <div className="p-6 bg-slate-800 rounded-xl shadow-lg border border-slate-700 flex flex-col">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-white">2. Run in Google Colab</h2>
          <button
            onClick={copyToClipboard}
            className={`text-xs px-3 py-1.5 rounded-md font-medium transition-colors ${
              copySuccess 
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50' 
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600 border border-slate-600'
            }`}
          >
            {copySuccess ? 'Copied!' : 'Copy Code'}
          </button>
        </div>
        
        <div className="relative flex-1 bg-slate-950 rounded-lg border border-slate-700 overflow-hidden group">
           <pre className="absolute inset-0 p-4 text-xs font-mono text-slate-400 overflow-auto whitespace-pre">
             {COLAB_SCRIPT}
           </pre>
        </div>
        
        <p className="mt-4 text-xs text-slate-500">
          Create a new cell in Colab, paste this code, and run it. The initial setup might take ~2 minutes.
          <br/>
          <strong>Note:</strong> You may need a free ngrok authtoken if the tunnel fails.
        </p>
      </div>
    </div>
  );
};
