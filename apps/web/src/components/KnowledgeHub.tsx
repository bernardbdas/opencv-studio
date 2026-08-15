import React, { useState, useEffect } from 'react';
import { OpenCVStudioClient } from '@opencv-studio/shared';
import { BookOpen, Search, Tag, Calendar, ShieldCheck, ChevronRight, FileText, Home } from 'lucide-react';
import { formatMathText } from '../utils/mathFormatter';

const api = new OpenCVStudioClient();

interface ArticleMeta {
  key: string;
  type: string;
  title: string;
  description: string;
  tags: string[];
  timestamp: string;
}

interface ArticleDetail {
  key: string;
  type: string;
  title: string;
  description: string;
  tags: string[];
  timestamp: string;
  content: string;
}

export function KnowledgeHub() {
  const [catalog, setCatalog] = useState<ArticleMeta[]>([]);
  const [activeArticle, setActiveArticle] = useState<ArticleDetail | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadCatalog();
  }, []);

  const loadCatalog = async () => {
    setLoading(true);
    try {
      const data = await api.getKnowledgeCatalog();
      setCatalog(data);
      if (data.length > 0) {
        // Load the README/index page if available, else first article
        const readme = data.find((a: ArticleMeta) => a.key === 'README' || a.type === 'index');
        const defaultKey = readme ? readme.key : data[0].key;
        await loadArticle(defaultKey);
      }
      setError(null);
    } catch (err: any) {
      setError('Failed to load knowledge catalog: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadArticle = async (key: string) => {
    setLoading(true);
    try {
      const data = await api.getKnowledgeArticle(key);
      setActiveArticle(data);
      setError(null);
    } catch (err: any) {
      setError(`Failed to load article '${key}': ` + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleWikiLinkClick = (key: string) => {
    loadArticle(key);
  };

  // Filter catalog based on search query
  const filteredCatalog = catalog.filter((item) => {
    const query = searchQuery.toLowerCase();
    return (
      item.title.toLowerCase().includes(query) ||
      item.description.toLowerCase().includes(query) ||
      item.tags.some((t) => t.toLowerCase().includes(query))
    );
  });

  return (
    <div className="space-y-6">
      {/* Catalog Title */}
      <div className="glass-panel p-6 rounded-2xl flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-3">
            <BookOpen className="w-7 h-7 text-cyan-400" />
            <span>Studio Knowledge Catalog</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1 leading-relaxed">
            Obsidian-powered internal wiki mapping traditional computer vision equations, deep learning structures, and YOLO architectures.
          </p>
        </div>
        <div className="flex items-center space-x-1.5 px-3 py-1 bg-slate-900 rounded-lg border border-slate-800 text-[10px] font-mono text-slate-400">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>Google OKF v0.2 Compliant</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Column: Sidebar Navigator */}
        <div className="lg:col-span-1 glass-panel p-5 rounded-2xl space-y-4 flex flex-col h-[calc(100vh-280px)] min-h-[500px]">
          {/* Search bar */}
          <div className="relative">
            <input
              type="text"
              placeholder="Search articles..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-slate-900 border border-slate-800 text-xs text-white rounded-xl placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 transition font-mono"
            />
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-3" />
          </div>

          {/* List items */}
          <div className="flex-1 overflow-y-auto space-y-1.5 pr-1 scrollbar-thin">
            {filteredCatalog.map((item) => {
              const isActive = activeArticle?.key === item.key;
              const isIndex = item.key === 'README' || item.type === 'index';
              return (
                <button
                  key={item.key}
                  onClick={() => loadArticle(item.key)}
                  className={`w-full p-3 rounded-xl border text-left flex items-start space-x-2.5 transition group ${
                    isActive
                      ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-300'
                      : 'border-slate-850 bg-slate-900/30 text-slate-400 hover:bg-slate-900/60'
                  }`}
                >
                  {isIndex ? (
                    <Home className={`w-4 h-4 mt-0.5 shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-400'}`} />
                  ) : (
                    <FileText className={`w-4 h-4 mt-0.5 shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-400'}`} />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="text-[11px] font-bold truncate leading-tight">
                      {item.title}
                    </div>
                    <div className="text-[9px] text-slate-500 truncate mt-0.5 leading-tight">
                      {item.description}
                    </div>
                  </div>
                  <ChevronRight className="w-3 h-3 text-slate-600 self-center shrink-0" />
                </button>
              );
            })}
            {filteredCatalog.length === 0 && (
              <div className="text-center py-8 text-xs text-slate-500 font-mono">
                No matching articles found.
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Article Reader View */}
        <div className="lg:col-span-3 glass-panel p-8 rounded-2xl overflow-y-auto h-[calc(100vh-280px)] min-h-[500px] scrollbar-thin flex flex-col">
          {error && (
            <div className="p-4 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-xl text-xs flex items-center space-x-2 mb-6">
              <span>{error}</span>
            </div>
          )}

          {activeArticle ? (
            <div className="space-y-6 flex-1">
              {/* Header Details */}
              <div className="pb-5 border-b border-slate-800 space-y-3">
                <h1 className="text-3xl font-extrabold text-white tracking-tight font-mono">
                  {activeArticle.title}
                </h1>
                <p className="text-xs text-slate-400 italic leading-relaxed">
                  {activeArticle.description}
                </p>
                <div className="flex flex-wrap gap-4 pt-1 items-center">
                  <div className="flex items-center space-x-1.5 text-[10px] text-slate-500 font-mono">
                    <Tag className="w-3.5 h-3.5 text-cyan-500" />
                    <span className="uppercase text-[9px] font-bold text-slate-400">Metadata Type:</span>
                    <span className="bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-cyan-300">{activeArticle.type}</span>
                  </div>
                  <div className="flex items-center space-x-1.5 text-[10px] text-slate-500 font-mono">
                    <Calendar className="w-3.5 h-3.5 text-pink-500" />
                    <span className="uppercase text-[9px] font-bold text-slate-400">Published:</span>
                    <span>{new Date(activeArticle.timestamp).toLocaleString()}</span>
                  </div>
                  {activeArticle.tags.length > 0 && (
                    <div className="flex items-center gap-1.5">
                      {activeArticle.tags.map((tag) => (
                        <span key={tag} className="text-[9px] bg-slate-900/60 border border-slate-800/80 px-2 py-0.5 rounded-full text-slate-450 font-mono">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Parsed Body */}
              <div className="article-body prose prose-invert max-w-none">
                {parseMarkdown(activeArticle.content, handleWikiLinkClick)}
              </div>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
              <BookOpen className="w-12 h-12 text-slate-600 mb-3 animate-pulse" />
              <div className="text-xs text-slate-500 font-mono">
                Select an article from the sidebar catalog to begin reading.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Interactive custom parser for OKF / Obsidian Markdown
function parseMarkdown(md: string, onLinkClick: (key: string) => void): React.ReactNode {
  const lines = md.split('\n');
  const elements: React.ReactNode[] = [];
  
  let inCodeBlock = false;
  let codeBlockLines: string[] = [];
  let codeLang = '';
  
  let inTable = false;
  let tableRows: string[][] = [];
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Code block detection
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        inCodeBlock = false;
        elements.push(
          <pre key={`code-${i}`} className="p-4 bg-slate-950/80 rounded-xl border border-slate-800/80 text-xs font-mono text-cyan-300 overflow-x-auto my-3.5 leading-relaxed scrollbar-thin">
            {codeBlockLines.join('\n')}
          </pre>
        );
        codeBlockLines = [];
      } else {
        inCodeBlock = true;
        codeLang = line.replace('```', '').trim();
      }
      continue;
    }
    
    if (inCodeBlock) {
      codeBlockLines.push(line);
      continue;
    }
    
    // Table detection
    if (line.trim().startsWith('|')) {
      inTable = true;
      const cols = line.split('|').map(c => c.trim()).slice(1, -1);
      // Skip delimiter lines (e.g., | :--- | --- |)
      if (!cols.every(c => c.startsWith(':') || c.startsWith('-'))) {
        tableRows.push(cols);
      }
      continue;
    } else if (inTable) {
      inTable = false;
      elements.push(
        <div key={`table-${i}`} className="overflow-x-auto my-4 rounded-xl border border-slate-800/80">
          <table className="min-w-full divide-y divide-slate-800">
            <thead className="bg-slate-900/60">
              <tr>
                {tableRows[0]?.map((col, idx) => (
                  <th key={idx} className="px-4 py-2.5 text-left text-[11px] font-bold text-slate-200 tracking-wider">
                    {renderInlineText(col, onLinkClick)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-850 bg-slate-950/20">
              {tableRows.slice(1).map((row, rowIdx) => (
                <tr key={rowIdx} className="hover:bg-slate-900/30 transition">
                  {row.map((col, colIdx) => (
                    <td key={colIdx} className="px-4 py-2.5 text-[11px] text-slate-400">
                      {renderInlineText(col, onLinkClick)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      tableRows = [];
    }
    
    const trimmed = line.trim();
    if (!trimmed) {
      // Empty line adds standard spacing
      elements.push(<div key={`space-${i}`} className="h-2" />);
      continue;
    }
    
    // Headers
    if (trimmed.startsWith('# ')) {
      elements.push(<h1 key={i} className="text-xl font-bold text-white mt-6 mb-3 tracking-tight font-mono">{renderInlineText(trimmed.slice(2), onLinkClick)}</h1>);
    } else if (trimmed.startsWith('## ')) {
      elements.push(<h2 key={i} className="text-[14px] font-bold text-cyan-400 mt-5 mb-2.5 border-b border-slate-800 pb-1.5 font-mono">{renderInlineText(trimmed.slice(3), onLinkClick)}</h2>);
    } else if (trimmed.startsWith('### ')) {
      elements.push(<h3 key={i} className="text-xs font-bold text-slate-200 mt-4 mb-2 font-mono">{renderInlineText(trimmed.slice(4), onLinkClick)}</h3>);
    }
    // Lists
    else if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
      elements.push(
        <li key={i} className="ml-6 list-disc text-xs text-slate-405 my-1 leading-relaxed">
          {renderInlineText(trimmed.slice(2), onLinkClick)}
        </li>
      );
    }
    // Horizontal rule
    else if (trimmed === '---') {
      elements.push(<hr key={i} className="border-slate-800 my-5" />);
    }
    // Paragraph text
    else {
      elements.push(<p key={i} className="text-xs text-slate-400 my-1 leading-relaxed">{renderInlineText(trimmed, onLinkClick)}</p>);
    }
  }
  
  return <div className="space-y-0.5">{elements}</div>;
}

function renderInlineText(text: string, onLinkClick: (key: string) => void): React.ReactNode[] {
  const tokens: React.ReactNode[] = [];
  let index = 0;
  
  // Match [[ObsidianLink]] or **BoldText**
  const regex = /(\[\[.*?\]\]|\*\*.*?\*\*)/g;
  let match;
  
  while ((match = regex.exec(text)) !== null) {
    if (match.index > index) {
      tokens.push(formatMathText(text.slice(index, match.index)));
    }
    
    const token = match[0];
    if (token.startsWith('[[')) {
      const linkKey = token.slice(2, -2).trim();
      tokens.push(
        <button
          key={match.index}
          onClick={() => onLinkClick(linkKey)}
          className="text-cyan-400 hover:text-cyan-300 font-bold underline transition inline-block align-baseline bg-cyan-955/20 px-1 py-0.5 rounded cursor-pointer"
        >
          {linkKey.replace(/_/g, ' ')}
        </button>
      );
    } else if (token.startsWith('**')) {
      const boldText = token.slice(2, -2);
      tokens.push(<strong key={match.index} className="font-bold text-slate-205">{boldText}</strong>);
    }
    
    index = regex.lastIndex;
  }
  
  if (index < text.length) {
    tokens.push(formatMathText(text.slice(index)));
  }
  
  return tokens;
}
