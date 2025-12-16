import { useState, useRef, useEffect } from 'react';
import SourceModal from './SourceModal';

function Chat({ selectedCollection }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState({});
  const [selectedSource, setSelectedSource] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const toggleSources = (messageIndex) => {
    setExpandedSources(prev => ({
      ...prev,
      [messageIndex]: !prev[messageIndex]
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const requestBody = { 
        question: userMessage, 
        top_k: 3, 
        use_hybrid: true
      };
      
      // Use selected collection if available
      if (selectedCollection) {
        requestBody.collection_name = selectedCollection;
      }

      const response = await fetch('/api/query/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Query failed');
      }

      const data = await response.json();
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        sources: data.sources || [],
        graphContext: data.graph_context,
        provenance: data.provenance,
        question: userMessage  // Store the question for highlighting
      }]);
    } catch (err) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Error: ${err.message}`,
        error: true
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 flex flex-col h-[600px]">
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center">
          <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
          Chat with your Knowledge Base
        </h2>
        
        {/* Collection Indicator */}
        {selectedCollection ? (
          <div className="mt-2 flex items-center text-sm text-gray-600 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2">
            <svg className="w-4 h-4 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className="font-medium">Querying:</span>
            <span className="ml-1 text-blue-700 font-mono text-xs">{selectedCollection}</span>
          </div>
        ) : (
          <div className="mt-2 flex items-center text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded-lg px-3 py-2">
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <span>No document selected. Select a document from the list or use default collection.</span>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto mb-4 space-y-4 scrollbar-thin pr-2">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-8">
            <svg className="mx-auto h-16 w-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <p className="text-lg">Start a conversation</p>
            <p className="text-sm mt-2">Ask questions about your uploaded documents</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`animate-fadeIn ${msg.role === 'user' ? 'flex justify-end' : ''}`}>
            <div className={`max-w-[80%] rounded-lg p-4 ${
              msg.role === 'user' 
                ? 'bg-blue-600 text-white' 
                : msg.error 
                  ? 'bg-red-50 border border-red-200 text-red-800'
                  : 'bg-gray-100 text-gray-800'
            }`}>
              <p className="whitespace-pre-wrap">{msg.content}</p>
              
              {msg.role === 'assistant' && !msg.error && msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-300">
                  <button
                    onClick={() => toggleSources(idx)}
                    className="flex items-center text-sm font-medium text-blue-600 hover:text-blue-800"
                  >
                    <svg className={`w-4 h-4 mr-1 transition-transform ${expandedSources[idx] ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                    {expandedSources[idx] ? 'Hide' : 'Show'} Sources ({msg.sources.length})
                  </button>
                  
                  {expandedSources[idx] && (
                    <div className="mt-2 space-y-2">
                      {msg.sources.map((source, sidx) => (
                        <div key={sidx} className="bg-white rounded-lg p-4 text-sm border-2 border-gray-200 hover:border-blue-300 transition-all">
                          {/* Source Header */}
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className={`px-2 py-1 text-xs rounded-full font-medium ${
                                source.type === 'graph' 
                                  ? 'bg-purple-100 text-purple-800' 
                                  : 'bg-green-100 text-green-800'
                              }`}>
                                {source.type === 'graph' ? '🔷 Graph' : '📄 Vector'}
                              </span>
                              {source.doc_title && (
                                <span className="text-xs text-gray-600 font-medium truncate max-w-xs">
                                  📚 {source.doc_title}
                                </span>
                              )}
                              {source.page && (
                                <span className="text-xs text-gray-500">
                                  pg. {source.page}
                                </span>
                              )}
                            </div>
                          </div>
                          
                          {/* Source Preview */}
                          <p className="text-gray-700 text-xs leading-relaxed mb-2 line-clamp-3">
                            {source.content.substring(0, 200)}...
                          </p>
                          
                          {/* Source Footer */}
                          {source.chunk_id && (
                            <div className="pt-2 border-t border-gray-100">
                              <span className="text-xs text-gray-400 font-mono">
                                ID: {source.chunk_id}
                              </span>
                            </div>
                          )}
                        </div>
                      ))}
                      
                      {msg.provenance && msg.provenance.node_ids && msg.provenance.node_ids.length > 0 && (
                        <div className="bg-purple-50 rounded-lg p-4 text-sm border-2 border-purple-200">
                          <h5 className="font-semibold text-purple-800 mb-2 flex items-center">
                            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                            </svg>
                            Knowledge Graph Nodes ({msg.provenance.node_ids.length})
                          </h5>
                          <div className="flex flex-wrap gap-2">
                            {msg.provenance.node_ids.map((nodeId, nidx) => {
                              const [entity, label] = nodeId.split('|');
                              return (
                                <div key={nidx} className="bg-white rounded px-3 py-2 border border-purple-200">
                                  <div className="font-medium text-purple-900 text-xs">{entity}</div>
                                  {label && (
                                    <div className="text-xs text-purple-600 mt-0.5">{label}</div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start animate-fadeIn">
            <div className="bg-gray-100 rounded-lg p-4 max-w-[80%]">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                </div>
                <span className="text-sm text-gray-600">Thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
          className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
        />
        <button
          type="submit"
          disabled={!input.trim() || loading}
          className={`px-6 py-3 rounded-lg font-medium text-white transition-all duration-200 ${
            !input.trim() || loading 
              ? 'bg-gray-400 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700 active:scale-95'
          }`}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </form>

      {/* Source Modal */}
      {selectedSource && (
        <SourceModal 
          source={selectedSource} 
          onClose={() => setSelectedSource(null)} 
        />
      )}
    </div>
  );
}

export default Chat;
