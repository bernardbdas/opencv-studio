import React, { useState } from 'react';
import { MessageSquare, Cpu, Send, Hash } from 'lucide-react';

export const AudioTextStudio: React.FC = () => {
  const [prompt, setPrompt] = useState('Explain how U-Net feature skip connections improve image segmentation accuracy.');
  const [llmOutput, setLlmOutput] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [embedText, setEmbedText] = useState('OpenCV Studio custom CUDA kernel acceleration');
  const [embedding, setEmbedding] = useState<number[] | null>(null);

  const runLLM = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/nlp/llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      const data = await res.json();
      setLlmOutput(data.response);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const runEmbed = async () => {
    try {
      const res = await fetch('/api/nlp/text-embedding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: embedText })
      });
      const data = await res.json();
      setEmbedding(data.embedding);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* LLM Lab */}
      <div className="glass-panel p-6 rounded-2xl space-y-6">
        <div className="flex items-center space-x-3">
          <MessageSquare className="w-6 h-6 text-cyan-400" />
          <h3 className="text-xl font-bold text-white">Generative AI LLM Lab</h3>
        </div>

        <div className="space-y-3">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            className="w-full bg-slate-900 border border-slate-800 focus:border-cyan-500 rounded-xl p-3 text-sm text-slate-200 focus:outline-none transition resize-none"
            placeholder="Type prompt..."
          />
          <button
            onClick={runLLM}
            disabled={loading}
            className="w-full py-2.5 bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white rounded-xl text-sm font-medium transition flex items-center justify-center space-x-2 disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
            <span>Generate LLM Inference</span>
          </button>
        </div>

        {llmOutput && (
          <div className="p-4 bg-slate-950/80 border border-cyan-500/20 rounded-xl text-sm text-slate-300">
            {llmOutput}
          </div>
        )}
      </div>

      {/* Embeddings Lab */}
      <div className="glass-panel p-6 rounded-2xl space-y-6">
        <div className="flex items-center space-x-3">
          <Hash className="w-6 h-6 text-cyan-400" />
          <h3 className="text-xl font-bold text-white">Vector Embedding Lab</h3>
        </div>

        <div className="space-y-3">
          <input
            type="text"
            value={embedText}
            onChange={(e) => setEmbedText(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 focus:border-cyan-500 rounded-xl px-4 py-2.5 text-sm text-slate-200 focus:outline-none transition"
          />
          <button
            onClick={runEmbed}
            className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-xl text-sm font-medium transition"
          >
            Generate Dense Embedding
          </button>
        </div>

        {embedding && (
          <div className="p-4 bg-slate-950/80 border border-slate-800 rounded-xl text-xs font-mono space-y-2">
            <div className="text-cyan-400 font-semibold">16-Dimensional Dense Vector:</div>
            <div className="text-slate-400 break-words">
              [{embedding.join(', ')}]
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
