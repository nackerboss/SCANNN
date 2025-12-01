import React, { useState } from 'react';
import { runBenchmark } from '../services/api';
import { BenchmarkResponse } from '../services/types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';

interface BenchmarkSectionProps {
  baseUrl: string;
}

export const BenchmarkSection: React.FC<BenchmarkSectionProps> = ({ baseUrl }) => {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleRun = async () => {
    setLoading(true);
    setError('');
    try {
      // Using 200 queries is a good balance between speed and statistical significance for a demo
      const result = await runBenchmark(baseUrl, 200); 
      setData(result);
    } catch (err) {
      setError('Benchmark failed. The server might be busy, disconnected, or OOM.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto">
      <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700 mb-8">
        <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Algorithm Benchmark</h2>
            <p className="text-slate-400 max-w-xl">
              Compare <strong>Brute Force</strong> (Exact k-NN) vs. <strong>ScaNN</strong> (Approximate k-NN) performance on 200 random test queries. 
              <br/><span className="text-xs text-slate-500">Metrics: Latency, Recall, Memory Usage, Data Compression.</span>
            </p>
          </div>
          <button
            onClick={handleRun}
            disabled={loading}
            className="shrink-0 px-6 py-3 bg-rose-600 hover:bg-rose-500 text-white font-bold rounded-lg transition-colors shadow-lg shadow-rose-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Running...
              </span>
            ) : 'Run Benchmark'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-900/20 border border-red-500/50 rounded-lg text-red-300 mb-6">
          {error}
        </div>
      )}

      {data && (
        <div className="animate-fade-in-up space-y-8">
          
          {/* Charts Row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Speed Chart */}
            <div className="bg-slate-800 p-5 rounded-xl border border-slate-700 flex flex-col h-80">
              <h3 className="text-base font-semibold text-white mb-4 text-center">Avg Search Time (ms) <span className="block text-[10px] font-normal text-slate-400">(Lower is better)</span></h3>
              <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.results} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="method" stroke="#94a3b8" fontSize={10} tick={{fill: '#e2e8f0'}} />
                    <YAxis stroke="#94a3b8" fontSize={10} tick={{fill: '#e2e8f0'}} />
                    <Tooltip 
                      cursor={{fill: '#334155', opacity: 0.2}}
                      contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#fff', fontSize: '12px' }}
                    />
                    <Bar dataKey="avg_ms_per_query" name="Time (ms)" radius={[4, 4, 0, 0]} barSize={40}>
                      {data.results.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.method.includes('ScaNN') ? '#818cf8' : '#f43f5e'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Recall Chart */}
            <div className="bg-slate-800 p-5 rounded-xl border border-slate-700 flex flex-col h-80">
              <h3 className="text-base font-semibold text-white mb-4 text-center">Accuracy (Recall@5) <span className="block text-[10px] font-normal text-slate-400">(Higher is better)</span></h3>
              <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.results} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="method" stroke="#94a3b8" fontSize={10} tick={{fill: '#e2e8f0'}} />
                    <YAxis domain={[0, 1.1]} ticks={[0, 0.5, 1]} stroke="#94a3b8" fontSize={10} tick={{fill: '#e2e8f0'}} tickFormatter={(v)=>`${v*100}%`}/>
                    <Tooltip 
                       cursor={{fill: '#334155', opacity: 0.2}}
                       contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#fff', fontSize: '12px' }}
                       formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Recall']}
                    />
                    <Bar dataKey="recall" name="Recall" radius={[4, 4, 0, 0]} barSize={40}>
                      {data.results.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.method.includes('ScaNN') ? '#818cf8' : '#f43f5e'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Memory Chart */}
            <div className="bg-slate-800 p-5 rounded-xl border border-slate-700 flex flex-col h-80">
              <h3 className="text-base font-semibold text-white mb-4 text-center">Memory Usage (MB) <span className="block text-[10px] font-normal text-slate-400">(Lower is better)</span></h3>
              <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.results} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="method" stroke="#94a3b8" fontSize={10} tick={{fill: '#e2e8f0'}} />
                    <YAxis stroke="#94a3b8" fontSize={10} tick={{fill: '#e2e8f0'}} />
                    <Tooltip 
                      cursor={{fill: '#334155', opacity: 0.2}}
                      contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#fff', fontSize: '12px' }}
                    />
                    <Bar dataKey="memory_mb" name="Memory (MB)" radius={[4, 4, 0, 0]} barSize={40}>
                      {data.results.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.method.includes('ScaNN') ? '#818cf8' : '#f43f5e'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Results Table */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <div className="p-4 bg-slate-900/50 border-b border-slate-700 flex justify-between items-center">
              <h3 className="font-bold text-white">Detailed Statistics</h3>
              <div className="text-sm">
                 <span className="text-slate-400">Data Compression Ratio: </span>
                 <span className="text-emerald-400 font-mono font-bold text-lg">{data.compression_ratio.toFixed(2)}x</span>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-700/30 text-slate-300 text-xs uppercase tracking-wider">
                    <th className="p-4 font-medium">Method</th>
                    <th className="p-4 font-medium">Total Time (s)</th>
                    <th className="p-4 font-medium">Avg Time (ms)</th>
                    <th className="p-4 font-medium">Recall@5</th>
                    <th className="p-4 font-medium">Memory (MB)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50 text-slate-200 text-sm">
                  {data.results.map((row) => (
                    <tr key={row.method} className="hover:bg-slate-700/20 transition-colors">
                      <td className="p-4 font-bold flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${row.method === 'ScaNN' ? 'bg-indigo-400' : 'bg-rose-400'}`}></span>
                        {row.method}
                      </td>
                      <td className="p-4 font-mono">{row.time_seconds.toFixed(4)}s</td>
                      <td className="p-4 font-mono">{row.avg_ms_per_query.toFixed(3)}ms</td>
                      <td className="p-4 font-mono">{(row.recall * 100).toFixed(2)}%</td>
                      <td className="p-4 font-mono">{row.memory_mb.toFixed(2)} MB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
             <div className="p-4 bg-slate-900/30 text-xs text-slate-500 border-t border-slate-700">
                <strong>Analysis:</strong> ScaNN achieves 
                <span className="text-emerald-400 mx-1">
                  {((data.results.find(r => r.method === 'Brute Force')?.time_seconds || 1) / (data.results.find(r => r.method === 'ScaNN')?.time_seconds || 1)).toFixed(1)}x
                </span> 
                faster search speed and 
                <span className="text-emerald-400 mx-1">{data.compression_ratio.toFixed(1)}x</span> 
                data compression compared to brute force, while maintaining 
                <span className="text-emerald-400 mx-1">{(data.results.find(r => r.method === 'ScaNN')?.recall || 0 * 100).toFixed(1)}%</span> 
                accuracy.
             </div>
          </div>

        </div>
      )}
    </div>
  );
};