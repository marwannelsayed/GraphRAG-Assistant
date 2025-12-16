import { useState, useEffect } from 'react';

function DocumentList({ onSelectDocument }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [deleting, setDeleting] = useState(null);
  const [showDeleteAllModal, setShowDeleteAllModal] = useState(false);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/documents/');
      if (!response.ok) {
        throw new Error('Failed to fetch documents');
      }
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching documents:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteDocument = async (collectionName, e) => {
    e.stopPropagation(); // Prevent document selection when clicking delete
    
    if (!confirm(`Are you sure you want to delete "${collectionName}"? This cannot be undone.`)) {
      return;
    }

    setDeleting(collectionName);
    try {
      const response = await fetch(`/api/documents/${encodeURIComponent(collectionName)}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete document');
      }

      // Refresh the document list
      await fetchDocuments();
      
      // If the deleted document was selected, clear selection
      if (selectedDoc === collectionName) {
        setSelectedDoc(null);
        if (onSelectDocument) {
          onSelectDocument(null);
        }
      }
    } catch (err) {
      alert(`Failed to delete document: ${err.message}`);
    } finally {
      setDeleting(null);
    }
  };

  const handleDeleteAll = async () => {
    setShowDeleteAllModal(false);
    setDeleting('all');
    
    try {
      const response = await fetch('/api/documents/', {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete all documents');
      }

      // Refresh the document list
      await fetchDocuments();
      
      // Clear selection
      setSelectedDoc(null);
      if (onSelectDocument) {
        onSelectDocument(null);
      }
    } catch (err) {
      alert(`Failed to delete all documents: ${err.message}`);
    } finally {
      setDeleting(null);
    }
  };

  const handleSelectDocument = (doc) => {
    setSelectedDoc(doc.collection_name);
    if (onSelectDocument) {
      onSelectDocument(doc.collection_name);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">📚 Uploaded Documents</h2>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Loading documents...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">📚 Uploaded Documents</h2>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-600">Error: {error}</p>
          <button
            onClick={fetchDocuments}
            className="mt-2 text-sm text-red-700 hover:text-red-800 underline"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800">📚 Uploaded Documents</h2>
        <div className="flex gap-2">
          <button
            onClick={fetchDocuments}
            disabled={loading || deleting !== null}
            className="text-sm px-3 py-1 text-blue-600 hover:text-blue-700 bg-blue-50 hover:bg-blue-100 rounded flex items-center gap-1 disabled:opacity-50"
            title="Refresh list"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>
          {documents.length > 0 && (
            <button
              onClick={() => setShowDeleteAllModal(true)}
              disabled={deleting !== null}
              className="text-sm px-3 py-1 text-red-600 hover:text-red-700 bg-red-50 hover:bg-red-100 rounded disabled:opacity-50"
              title="Delete all documents"
            >
              🗑️ Delete All
            </button>
          )}
        </div>
      </div>

      {documents.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <svg className="w-16 h-16 mx-auto mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="text-sm">No documents uploaded yet.</p>
          <p className="text-xs mt-1">Upload a document to get started.</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {documents.map((doc, index) => (
            <div
              key={index}
              onClick={() => handleSelectDocument(doc)}
              className={`p-4 border rounded-lg cursor-pointer transition-all hover:shadow-md ${
                selectedDoc === doc.collection_name
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300'
              } ${deleting === doc.collection_name ? 'opacity-50' : ''}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <svg className="w-5 h-5 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                    </svg>
                    <h3 className="font-medium text-gray-900 truncate" title={doc.original_filename}>
                      {doc.original_filename}
                    </h3>
                  </div>
                  
                  <div className="flex items-center gap-3 text-xs text-gray-500 mt-2">
                    <span className="flex items-center gap-1">
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                      </svg>
                      {doc.collection_name}
                    </span>
                    <span className="flex items-center gap-1">
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                      {doc.chunk_count} chunks
                    </span>
                  </div>
                </div>
                
                <div className="ml-2 flex-shrink-0 flex items-center gap-2">
                  {selectedDoc === doc.collection_name && (
                    <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  )}
                  <button
                    onClick={(e) => handleDeleteDocument(doc.collection_name, e)}
                    disabled={deleting !== null}
                    className="p-1.5 text-red-600 hover:bg-red-50 rounded disabled:opacity-50"
                    title="Delete this document"
                  >
                    {deleting === doc.collection_name ? (
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {documents.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-500 text-center">
            {documents.length} document{documents.length !== 1 ? 's' : ''} total
            {selectedDoc && ' • Click a document to use it for queries'}
          </p>
        </div>
      )}

      {/* Delete All Confirmation Modal */}
      {showDeleteAllModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowDeleteAllModal(false)}>
          <div className="bg-white rounded-lg p-6 max-w-md mx-4 shadow-xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-xl font-bold text-red-600 mb-4">⚠️ Delete All Documents?</h3>
            <p className="text-gray-700 mb-6">
              Are you sure you want to delete <strong>all {documents.length} documents</strong>? 
              This will remove all vector embeddings and graph data. This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowDeleteAllModal(false)}
                disabled={deleting !== null}
                className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteAll}
                disabled={deleting !== null}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 flex items-center gap-2"
              >
                {deleting === 'all' && (
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                )}
                Yes, Delete All
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default DocumentList;
