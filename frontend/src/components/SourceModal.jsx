import { useState, useEffect, useRef } from 'react';

function SourceModal({ source, onClose }) {
  const [loading, setLoading] = useState(false);
  const [fullSource, setFullSource] = useState(null);
  const [error, setError] = useState(null);
  const contentRef = useRef(null);

  // Fetch full source on mount if chunk_id is available
  useEffect(() => {
    if (source.chunk_id) {
      fetchFullSource(source.chunk_id);
    }
  }, [source.chunk_id]);

  // Scroll to highlighted text after render
  useEffect(() => {
    if (contentRef.current && !loading) {
      const mark = contentRef.current.querySelector('mark');
      if (mark) {
        // Wait a bit for the layout to settle
        setTimeout(() => {
          mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
      }
    }
  }, [fullSource, loading]);

  const fetchFullSource = async (chunkId) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/source/${chunkId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch source');
      }
      const data = await response.json();
      setFullSource(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const highlightSearchTerm = (text, question, answer) => {
    if (!text) return text;
    
    // Extract key terms from the question to find in the text
    let searchTerms = [];
    
    if (question) {
      // Remove common question words and extract key terms
      const cleanQuestion = question
        .toLowerCase()
        .replace(/^(what|who|where|when|why|how|is|are|was|were|does|do|did|the|a|an)\s+/gi, '')
        .replace(/\?/g, '');
      
      // Split into words and filter significant ones
      const questionWords = cleanQuestion.split(/\s+/).filter(w => w.length > 3);
      searchTerms.push(...questionWords);
    }
    
    if (answer) {
      // Extract key noun phrases from the answer (words with 5+ chars)
      const answerWords = answer
        .split(/\s+/)
        .filter(w => w.length >= 5 && !/^(this|that|these|those|which|where|there|their|would|could|should)$/i.test(w));
      searchTerms.push(...answerWords.slice(0, 5)); // Take first 5 significant words
    }
    
    if (searchTerms.length === 0) {
      return text;
    }
    
    // Find the section of text that contains the most search terms
    const lines = text.split('\n');
    let bestSection = { start: 0, end: 0, score: 0 };
    
    // Use a sliding window to find the most relevant section
    const windowSize = 8; // Look at 8 lines at a time
    for (let i = 0; i < lines.length - windowSize + 1; i++) {
      const window = lines.slice(i, i + windowSize).join('\n').toLowerCase();
      const score = searchTerms.filter(term => window.includes(term.toLowerCase())).length;
      
      if (score > bestSection.score) {
        bestSection = { start: i, end: i + windowSize, score };
      }
    }
    
    // If we found a good section, highlight it
    if (bestSection.score > 0) {
      const beforeLines = lines.slice(0, bestSection.start);
      const highlightLines = lines.slice(bestSection.start, bestSection.end);
      const afterLines = lines.slice(bestSection.end);
      
      const before = beforeLines.join('\n');
      const highlight = highlightLines.join('\n');
      const after = afterLines.join('\n');
      
      return `${before}<mark class="bg-yellow-300 font-semibold px-1 rounded block py-1 my-1">${highlight}</mark>${after}`;
    }
    
    // Fallback: highlight individual matching terms
    let highlightedText = text;
    searchTerms.forEach(term => {
      if (term.length > 4) {
        const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedTerm})`, 'gi');
        highlightedText = highlightedText.replace(regex, '<mark class="bg-yellow-300 font-semibold">$1</mark>');
      }
    });
    
    return highlightedText;
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 animate-fadeIn">
      <div className="bg-white rounded-lg shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-4 flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold">Source Document</h3>
            {source.doc_title && (
              <p className="text-sm text-blue-100 mt-1">
                {source.doc_title} {source.page && `• Page ${source.page}`}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-white hover:bg-white hover:bg-opacity-20 rounded-full p-2 transition-all"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Source Type Badge */}
          <div className="flex items-center gap-2">
            <span className={`px-3 py-1 text-sm font-medium rounded-full ${
              source.type === 'graph' 
                ? 'bg-purple-100 text-purple-800' 
                : 'bg-green-100 text-green-800'
            }`}>
              {source.type === 'graph' ? '🔷 Graph Source' : '📄 Vector Source'}
            </span>
            {source.chunk_id && (
              <span className="px-3 py-1 text-xs bg-gray-100 text-gray-600 rounded-full font-mono">
                ID: {source.chunk_id}
              </span>
            )}
          </div>

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800">
                <span className="font-semibold">Error:</span> {error}
              </p>
            </div>
          )}

          {/* Full Text */}
          {!loading && !error && (
            <>
              <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Full Text
                  {(source.question || source.answer) && (
                    <span className="ml-2 text-xs text-yellow-600 bg-yellow-50 px-2 py-1 rounded-full">
                      📍 Relevant section highlighted
                    </span>
                  )}
                </h4>
                <div 
                  ref={contentRef}
                  className="text-gray-800 leading-relaxed whitespace-pre-wrap"
                  dangerouslySetInnerHTML={{
                    __html: highlightSearchTerm(
                      fullSource?.text || source.content,
                      source.question,
                      source.answer
                    )
                  }}
                />
              </div>

              {/* Graph Entities */}
              {fullSource?.graph_entities && fullSource.graph_entities.length > 0 && (
                <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                  <h4 className="text-sm font-semibold text-purple-800 mb-3 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                    </svg>
                    Entities Mentioned ({fullSource.graph_entities.length})
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {fullSource.graph_entities.map((entity, idx) => (
                      <div key={idx} className="bg-white rounded px-3 py-2 border border-purple-200">
                        <div className="font-medium text-purple-900">{entity.text}</div>
                        <div className="text-xs text-purple-600 mt-0.5">{entity.label}</div>
                        {entity.description && (
                          <div className="text-xs text-gray-600 mt-1">{entity.description}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Graph Relations */}
              {fullSource?.graph_relations && fullSource.graph_relations.length > 0 && (
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <h4 className="text-sm font-semibold text-blue-800 mb-3 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Relationships ({fullSource.graph_relations.length})
                  </h4>
                  <div className="space-y-2">
                    {fullSource.graph_relations.map((rel, idx) => (
                      <div key={idx} className="bg-white rounded p-3 border border-blue-200 text-sm">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-blue-900">{rel.subject}</span>
                          <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded">
                            {rel.relation}
                          </span>
                          <span className="font-medium text-blue-900">{rel.object}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Metadata */}
              {source.metadata && Object.keys(source.metadata).length > 0 && (
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Metadata
                  </h4>
                  <dl className="grid grid-cols-2 gap-2 text-sm">
                    {Object.entries(source.metadata).map(([key, value]) => (
                      <div key={key} className="bg-white rounded px-3 py-2 border border-gray-100">
                        <dt className="text-xs text-gray-500 font-medium">{key}</dt>
                        <dd className="text-gray-800 mt-0.5 font-mono text-xs truncate">{String(value)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 flex justify-end border-t border-gray-200">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default SourceModal;
