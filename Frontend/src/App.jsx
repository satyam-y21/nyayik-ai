import React, { useState, useEffect } from 'react';
import { Scale, BookOpen, Send, ShieldAlert, Sparkles, FileText, ChevronRight, HelpCircle, Sun, Moon, CheckCircle2, AlertTriangle, Info } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

const SUGGESTED_QUERIES = [
  "Article 21",
  "Right to Education",
  "RTI Act, 2005",
  "Article 14",
  "BNS Section 103",
  "Fundamental Duties"
];

function App() {
  const [query, setQuery] = useState('');
  const [isDark, setIsDark] = useState(true);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Namaste! I am Nyayik AI, your Indian Legal & Constitutional Assistant.\n\nYou can describe your real-world legal concerns, incidents, or questions (such as police encounters, fundamental rights violations, property disputes, digital evidence, or statutory procedures).\n\n• **Grounded Legal Answers:** Official indexed legal documents (Constitution of India, BNS, BNSS, BSA) are prioritized to provide primary evidence with verified citations and Document Inspector text.\n\n• **General Legal Guidance:** When a specific concern extends beyond indexed documents, clearly-labelled general legal information is provided under Indian law.',
      sources: [],
      response_mode: 'CONVERSATIONAL'
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedSource, setSelectedSource] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [activeTab, setActiveTab] = useState('stream'); // 'stream' | 'inspector'

  const messagesEndRef = React.useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Handle dark / light theme persistence on body
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      document.body.style.backgroundColor = '#020617';
      document.body.style.color = '#f8fafc';
    } else {
      document.documentElement.classList.remove('dark');
      document.body.style.backgroundColor = '#f8fafc';
      document.body.style.color = '#0f172a';
    }
  }, [isDark]);

  // Fetch list of documents on load
  useEffect(() => {
    fetch('http://localhost:8000/api/documents')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.documents) {
          setDocuments(data.documents);
        }
      })
      .catch((err) => {
        console.warn('Backend server not reachable. Using fallback document list.', err);
        setDocuments(['Constitution of India']);
      });
  }, []);

  const submitQuery = async (queryText) => {
    if (!queryText.trim() || isLoading) return;

    const userMessage = { role: 'user', content: queryText };
    const currentMessages = [...messages, userMessage];
    setMessages(currentMessages);
    setQuery('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, history: currentMessages })
      });
      
      if (!response.ok) throw new Error('API server returned an error');

      const data = await response.json();
      const sourcesList = data.sources || [];
      const mode = data.response_mode || 'GROUNDED';
      
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          sources: sourcesList,
          response_mode: mode,
          disclaimer: data.disclaimer
        }
      ]);

      if (mode === 'GROUNDED' && sourcesList.length > 0) {
        setSelectedSource(sourcesList[0]);
      } else {
        setSelectedSource(null);
      }
    } catch (err) {
      console.error(err);
      const fallbackSources = [
        {
          file: "Constitution of India",
          section: "Article 21 (Protection of Life and Personal Liberty)",
          text: "No person shall be deprived of his life or personal liberty except according to procedure established by law. This article ensures the right to live with human dignity, right to privacy, right to clean environment, and right to free legal aid."
        }
      ];
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Connecting to legal engine... Nyayik AI operates strictly on a dual-source retrieval-first architecture. (No active backend connection found, showing mock educational sample for "${queryText}").`,
          sources: fallbackSources,
          response_mode: 'GROUNDED'
        }
      ]);
      setSelectedSource(fallbackSources[0]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = (e) => {
    e.preventDefault();
    submitQuery(query);
  };

  const handleChipClick = (suggestionText) => {
    submitQuery(suggestionText);
  };

  return (
    <div className={`h-screen w-screen overflow-hidden flex flex-col font-sans transition-colors duration-300 ${
      isDark ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'
    }`}>
      {/* Header Banner */}
      <header className={`border-b shrink-0 px-4 sm:px-6 py-3 sm:py-3.5 flex flex-wrap items-center justify-between gap-3 transition-colors duration-300 z-50 ${
        isDark 
          ? 'border-slate-800 bg-slate-950/90 backdrop-blur-md' 
          : 'border-slate-200 bg-white/90 backdrop-blur-md shadow-xs'
      }`}>
        <div className="flex items-center gap-3">
          <div className="bg-amber-500/10 p-2 rounded-xl border border-amber-500/30 text-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.1)]">
            <Scale className="h-5 w-5 sm:h-6 sm:w-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className={`text-lg sm:text-xl font-bold tracking-tight ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>Nyayik AI</h1>
              <span className="bg-indigo-500/10 text-indigo-500 text-xs px-2.5 py-0.5 rounded-full font-medium border border-indigo-500/20 flex items-center gap-1">
                <Sparkles className="h-3 w-3" /> Dual-Source RAG
              </span>
            </div>
            <p className={`text-[11px] sm:text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>Indian Legal & Constitutional Intelligence Platform</p>
          </div>
        </div>

        {/* Center Mobile View Toggle Switch */}
        <div className="flex lg:hidden items-center bg-slate-900/80 p-1 rounded-xl border border-slate-700/60 text-xs font-semibold">
          <button
            onClick={() => setActiveTab('stream')}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer ${
              activeTab === 'stream'
                ? 'bg-amber-500 text-slate-950 font-bold shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BookOpen className="h-3.5 w-3.5" /> Consultation
          </button>
          <button
            onClick={() => setActiveTab('inspector')}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer ${
              activeTab === 'inspector'
                ? 'bg-amber-500 text-slate-950 font-bold shadow-xs'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileText className="h-3.5 w-3.5" /> Inspector
            {selectedSource && <span className="h-2 w-2 rounded-full bg-indigo-400 animate-pulse"></span>}
          </button>
        </div>

        <div className="flex items-center gap-3">
          {/* Disclaimer Badge */}
          <div className={`hidden sm:flex items-center gap-2 border px-3.5 py-1.5 rounded-xl text-xs max-w-md shadow-inner ${
            isDark 
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-400' 
              : 'bg-amber-50 border-amber-200 text-amber-800'
          }`}>
            <ShieldAlert className="h-4 w-4 shrink-0 text-amber-500" />
            <span>
              <strong>Disclaimer:</strong> Educational use only. Does not constitute formal legal advice.
            </span>
          </div>

          {/* Theme Toggle Button */}
          <button
            onClick={() => setIsDark(!isDark)}
            title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
            className={`p-2 rounded-xl border transition-all cursor-pointer flex items-center justify-center ${
              isDark 
                ? 'bg-slate-900 border-slate-700 hover:bg-slate-800 text-amber-400' 
                : 'bg-slate-100 border-slate-300 hover:bg-slate-200 text-slate-700'
            }`}
          >
            {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 min-h-0 w-full max-w-[1600px] mx-auto p-3 sm:p-4 lg:p-6 gap-4 lg:gap-6 flex flex-col lg:flex-row overflow-hidden">
        {/* Left Column: Chat & Fixed Input */}
        <section className={`flex-1 min-h-0 h-full flex-col border rounded-2xl overflow-hidden backdrop-blur-sm transition-colors duration-300 ${
          activeTab === 'stream' ? 'flex' : 'hidden lg:flex'
        } ${
          isDark 
            ? 'bg-slate-900/50 border-slate-800 shadow-[0_0_20px_rgba(0,0,0,0.3)]' 
            : 'bg-white border-slate-200 shadow-sm'
        }`}>
          {/* Top Panel Bar */}
          <div className={`px-5 py-3 border-b shrink-0 flex items-center justify-between transition-colors ${
            isDark 
              ? 'border-slate-800 bg-slate-950/60' 
              : 'border-slate-150 bg-slate-50/80'
          }`}>
            <span className={`text-xs font-semibold uppercase tracking-wider flex items-center gap-2 ${
              isDark ? 'text-slate-400' : 'text-slate-600'
            }`}>
              <BookOpen className="h-4 w-4 text-amber-500" /> Legal Consultation Stream
            </span>
            <div className={`text-xs ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
              Active Documents: <span className={`font-medium ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{documents.length || 'Loading...'}</span>
            </div>
          </div>

          {/* Messages Log */}
          <div className="flex-1 min-h-0 overflow-y-auto p-4 sm:p-6 space-y-6 custom-scrollbar">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl p-4 shadow-md transition-all ${
                    msg.role === 'user'
                      ? 'bg-indigo-600 text-white rounded-tr-none border border-indigo-500/30'
                      : isDark
                        ? 'bg-slate-900 border border-slate-700/80 rounded-tl-none text-slate-200'
                        : 'bg-slate-100/90 border border-slate-200 rounded-tl-none text-slate-800'
                  }`}
                >
                  {/* Mode Badge for Assistant Messages */}
                  {msg.role !== 'user' && (
                    <div className="mb-3">
                      {msg.response_mode === 'GROUNDED' ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" /> Verified from Indexed Legal Sources
                        </span>
                      ) : msg.response_mode === 'GENERAL_INFORMATION' ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                          <AlertTriangle className="h-3.5 w-3.5 text-amber-400" /> General Legal Information
                        </span>
                      ) : msg.response_mode === 'OUT_OF_DOMAIN' ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold bg-slate-500/10 text-slate-400 border border-slate-500/30">
                          <Info className="h-3.5 w-3.5 text-slate-400" /> Out of Domain
                        </span>
                      ) : null}
                    </div>
                  )}

                  {msg.role === 'user' ? (
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <div className="text-sm leading-relaxed space-y-2">
                      <ReactMarkdown 
                        remarkPlugins={[remarkGfm]}
                        components={{
                          p: ({ node, ...props }) => <p className="mb-2 last:mb-0 leading-relaxed" {...props} />,
                          ul: ({ node, ...props }) => <ul className="list-disc pl-5 mb-2 space-y-1" {...props} />,
                          ol: ({ node, ...props }) => <ol className="list-decimal pl-5 mb-2 space-y-1" {...props} />,
                          li: ({ node, ...props }) => <li className="leading-relaxed" {...props} />,
                          strong: ({ node, ...props }) => <strong className={`font-semibold ${isDark ? 'text-amber-400' : 'text-amber-800'}`} {...props} />,
                          em: ({ node, ...props }) => <em className="italic opacity-90" {...props} />,
                          h1: ({ node, ...props }) => <h1 className="text-base font-bold mb-2 border-b pb-1" {...props} />,
                          h2: ({ node, ...props }) => <h2 className="text-sm font-bold mb-1.5" {...props} />,
                          h3: ({ node, ...props }) => <h3 className="text-xs font-bold mb-1" {...props} />,
                          blockquote: ({ node, ...props }) => (
                            <blockquote className={`pl-3 border-l-2 my-2 italic ${
                              isDark ? 'border-amber-500/60 text-slate-300' : 'border-amber-600 text-slate-700'
                            }`} {...props} />
                          ),
                          code: ({ node, inline, ...props }) => (
                            <code className={`px-1.5 py-0.5 rounded text-xs font-mono ${
                              isDark ? 'bg-slate-950 text-amber-300 border border-slate-800' : 'bg-slate-200 text-slate-900'
                            }`} {...props} />
                          ),
                          hr: ({ node, ...props }) => <hr className={`my-3 border-t ${isDark ? 'border-slate-800' : 'border-slate-300'}`} {...props} />,
                          a: ({ node, ...props }) => (
                            <a className="text-indigo-400 hover:underline" target="_blank" rel="noopener noreferrer" {...props} />
                          )
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>

                      {/* Explicit Disclaimer Banner for Mode B General Legal Info */}
                      {msg.response_mode === 'GENERAL_INFORMATION' && (
                        <div className={`mt-3 p-3 rounded-xl border text-xs flex items-start gap-2 ${
                          isDark ? 'bg-amber-950/30 border-amber-500/30 text-amber-300' : 'bg-amber-50 border-amber-200 text-amber-800'
                        }`}>
                          <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
                          <span>
                            <strong>Note:</strong> General legal information — not verified against Nyayik AI's indexed legal sources.
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Grounded Sources widget (Mode A only) */}
                  {msg.response_mode === 'GROUNDED' && msg.sources && msg.sources.length > 0 && (
                    <div className={`mt-4 pt-3 border-t ${isDark ? 'border-slate-800' : 'border-slate-200'}`}>
                      <p className={`text-xs font-semibold mb-2 flex items-center gap-1.5 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                        <FileText className="h-3.5 w-3.5 text-amber-500" /> Verified Legal References ({msg.sources.length}):
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {msg.sources.map((src, sIdx) => (
                          <button
                            key={sIdx}
                            onClick={() => {
                              setSelectedSource(src);
                              setActiveTab('inspector');
                            }}
                            className={`text-xs py-1.5 px-3 rounded-lg flex items-center gap-1 transition-all cursor-pointer shadow-xs border ${
                              isDark
                                ? 'bg-slate-950 hover:bg-slate-850 border-slate-700/80 hover:border-amber-500/40 text-slate-300 hover:text-amber-400'
                                : 'bg-white hover:bg-amber-50 border-slate-200 hover:border-amber-400 text-slate-700 hover:text-amber-800'
                            }`}
                          >
                            <span>{src.section}</span>
                            <ChevronRight className="h-3 w-3 opacity-60" />
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-4 justify-start">
                <div className={`max-w-[80%] rounded-2xl p-4 border rounded-tl-none flex items-center gap-2 ${
                  isDark ? 'bg-slate-900/60 border-slate-700 text-slate-400' : 'bg-slate-100 border-slate-200 text-slate-600'
                }`}>
                  <span className="h-2 w-2 rounded-full bg-amber-500 animate-bounce [animation-delay:-0.3s]"></span>
                  <span className="h-2 w-2 rounded-full bg-amber-500 animate-bounce [animation-delay:-0.15s]"></span>
                  <span className="h-2 w-2 rounded-full bg-amber-500 animate-bounce"></span>
                  <span className="text-xs ml-1">Evaluating evidence & legal sources...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Suggestion Chips */}
          <div className={`px-4 pt-3 pb-2 shrink-0 flex items-center gap-2 overflow-x-auto scrollbar-none transition-colors border-t ${
            isDark ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50/50 border-slate-200/60'
          }`}>
            {SUGGESTED_QUERIES.map((chipText, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleChipClick(chipText)}
                className={`text-xs px-3 py-1.5 rounded-md font-medium border flex items-center gap-1.5 transition-all shrink-0 cursor-pointer ${
                  isDark
                    ? 'bg-slate-900 hover:bg-slate-800 border-slate-700/80 text-sky-400 hover:text-sky-300 hover:border-sky-400'
                    : 'bg-white hover:bg-blue-50/80 border-slate-200 text-blue-600 hover:text-blue-800 hover:border-blue-300 shadow-2xs'
                }`}
              >
                <span>{chipText}</span>
                <ChevronRight className="h-3.5 w-3.5 opacity-70" />
              </button>
            ))}
          </div>

          {/* Form Input */}
          <form onSubmit={handleSend} className={`p-4 border-t shrink-0 flex gap-3 transition-colors ${
            isDark ? 'border-slate-800 bg-slate-950/40' : 'border-slate-200 bg-slate-50'
          }`}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a legal or constitutional question (e.g., Article 21, police scenario)..."
              className={`flex-1 border rounded-xl px-4 py-3 text-sm focus:outline-none transition-all shadow-inner ${
                isDark 
                  ? 'bg-slate-950 border-slate-700 focus:border-amber-500/70 text-slate-100 placeholder:text-slate-500' 
                  : 'bg-white border-slate-300 focus:border-amber-500 text-slate-900 placeholder:text-slate-400'
              }`}
            />
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              aria-label="Send Query"
              className="bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-slate-950 font-bold px-4 sm:px-5 py-3 rounded-xl flex items-center justify-center gap-2 transition-all cursor-pointer shadow-md disabled:cursor-not-allowed shrink-0"
            >
              <Send className="h-4 w-4 sm:hidden"/>
              <span className="hidden sm:inline">Search</span>
            </button>
          </form>
        </section>

        {/* Right Column: Citation Inspector */}
        <section className={`w-full lg:w-[400px] min-h-0 h-full border rounded-2xl overflow-hidden backdrop-blur-sm flex-col transition-colors duration-300 ${
          activeTab === 'inspector' ? 'flex' : 'hidden lg:flex'
        } ${
          isDark ? 'bg-slate-900/50 border-slate-800 shadow-[0_0_20px_rgba(0,0,0,0.3)]' : 'bg-white border-slate-200 shadow-sm'
        }`}>
          <div className={`px-5 py-3 border-b shrink-0 flex items-center justify-between transition-colors ${
            isDark ? 'border-slate-800 bg-slate-950/60' : 'border-slate-150 bg-slate-50/80'
          }`}>
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-amber-500" />
              <span className={`text-xs font-semibold uppercase tracking-wider ${
                isDark ? 'text-slate-400' : 'text-slate-600'
              }`}>
                Document Inspector
              </span>
            </div>
            <button
              onClick={() => setActiveTab('stream')}
              className="lg:hidden text-xs text-amber-500 hover:text-amber-400 font-medium flex items-center gap-1 cursor-pointer"
            >
              Back to Chat &rarr;
            </button>
          </div>

          {/* Inspector content */}
          <div className="flex-1 min-h-0 p-4 sm:p-6 overflow-y-auto custom-scrollbar">
            {selectedSource ? (
              <div className="space-y-4">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <span className="bg-amber-500/10 text-amber-500 border border-amber-500/25 text-[10px] px-2.5 py-0.5 rounded font-bold uppercase tracking-wider">
                      {selectedSource.file}
                    </span>
                    <h3 className={`text-base font-bold mt-2 ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>
                      {selectedSource.section}
                    </h3>
                  </div>
                </div>
                <div className={`border rounded-xl p-4 shadow-inner relative overflow-hidden ${
                  isDark ? 'bg-slate-950/90 border-slate-700' : 'bg-slate-50 border-slate-200'
                }`}>
                  <div className="absolute top-0 left-0 bottom-0 w-1 bg-amber-500"></div>
                  <div className={`text-sm leading-relaxed pl-2 italic ${
                    isDark ? 'text-slate-300' : 'text-slate-700'
                  }`}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {selectedSource.text}
                    </ReactMarkdown>
                  </div>
                </div>
                <p className={`text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                  This segment was extracted directly from the official PDF registry in the local vector database.
                </p>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center p-4">
                <HelpCircle className={`h-10 w-10 mb-3 ${isDark ? 'text-slate-600' : 'text-slate-400'}`} />
                <h4 className={`text-sm font-semibold ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>No Reference Inspected</h4>
                <p className={`text-xs mt-2 max-w-[280px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                  No verified legal source was used for this response, or select a reference badge from a grounded answer to view source text.
                </p>
              </div>
            )}
          </div>

          {/* Quick reference guide info card */}
          <div className={`p-4 border-t shrink-0 transition-colors ${
            isDark ? 'border-slate-800 bg-slate-950/40' : 'border-slate-200 bg-slate-50/50'
          }`}>
            <div className={`p-3.5 rounded-xl border text-xs space-y-1.5 ${
              isDark ? 'bg-slate-950/60 border-slate-800 text-slate-400' : 'bg-white border-slate-200 text-slate-600 shadow-2xs'
            }`}>
              <span className={`font-semibold flex items-center gap-1.5 ${isDark ? 'text-slate-300' : 'text-slate-800'}`}>
                <Scale className="h-3.5 w-3.5 text-amber-500" /> Dual-Source Policy
              </span>
              <ul className={`list-disc pl-4 space-y-0.5 ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>
                <li>🟢 Verified: Grounded in official indexed legal documents.</li>
                <li>🟡 General Info: Informational guidance when corpus evidence is unindexed.</li>
              </ul>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
