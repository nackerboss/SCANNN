import React, { useState } from 'react';
import { SearchResponse } from '../services/types';
import { searchArticles } from '../services/api';

interface SearchSectionProps {
  baseUrl: string;
}

export const SearchSection: React.FC<SearchSectionProps> = ({ baseUrl }) => {
  const [query, setQuery] = useState('');
  const [k, setK] = useState(5);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError('');
    try {
      const data = await searchArticles(baseUrl, query, k);
      setResults(data);
    } catch (err) {
      setError('Search failed. Check connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700 mb-8">
        <form onSubmit={handleSearch} className="flex flex-col gap-4">
          <div className="flex gap-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter a search query (e.g., 'Latest technological advancements in AI')"
              className="flex-1 px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-lg transition-colors shadow-lg shadow-indigo-500/20 disabled:opacity-50"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>
          
          <div className="flex items-center gap-4 text-slate-300 text-sm">
            <label>Neighbors (k): <span className="font-mono font-bold text-indigo-400">{k}</span></label>
            <input
              type="range"
              min="1"
              max="50"
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-48 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>
        </form>
      </div>

      {error && (
        <div className="p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-300 mb-6">
          {error}
        </div>
      )}

      {results && (
        <div className="space-y-4">
          <div className="flex justify-between items-end mb-2">
            <h3 className="text-xl font-bold text-white">Results</h3>
            <span className="text-xs text-slate-400 font-mono">
              Took {results.search_time_ms.toFixed(2)}ms
            </span>
          </div>
          
          {results.results.map((item) => (
            <div 
              key={item.rank}
              className="bg-slate-800/50 border border-slate-700 p-4 rounded-lg hover:border-indigo-500/50 transition-colors group"
            >
              <div className="flex justify-between items-start gap-4">
                <div className="flex gap-4">
                  <div className="flex flex-col items-center justify-center w-12 h-12 bg-slate-900 rounded-full border border-slate-700 shrink-0 group-hover:border-indigo-500/50 group-hover:text-indigo-400 transition-colors text-slate-400 font-bold">
                    #{item.rank}
                  </div>
                  <div>
                    <p className="text-slate-200 leading-relaxed">{item.text}</p>
                    <p className="text-xs text-slate-500 mt-2">Dataset Index: {item.dataset_index}</p>
                  </div>
                </div>
                <div className="shrink-0 text-right">
                   <div className="text-sm font-mono text-emerald-400 font-bold">
                    {(item.similarity * 100).toFixed(1)}%
                   </div>
                   <div className="text-xs text-slate-500">Match</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};