import { useState } from 'react';
import Upload from './components/Upload';
import Chat from './components/Chat';
import DocumentList from './components/DocumentList';

function App() {
  const [uploadKey, setUploadKey] = useState(0);
  const [selectedCollection, setSelectedCollection] = useState(null);
  const [documentsKey, setDocumentsKey] = useState(0);

  const handleUploadSuccess = (data) => {
    console.log('Upload successful:', data);
    // Refresh document list after successful upload
    setDocumentsKey(prev => prev + 1);
    // Auto-select the newly uploaded document
    if (data.collection_name) {
      setSelectedCollection(data.collection_name);
    }
    setUploadKey(prev => prev + 1);
  };

  const handleSelectDocument = (collectionName) => {
    console.log('Selected document collection:', collectionName);
    setSelectedCollection(collectionName);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center justify-center mb-2">
            <svg className="w-10 h-10 text-blue-600 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
            </svg>
            <h1 className="text-4xl font-bold text-gray-800">
              Agentic HybridRAG Knowledge Engine
            </h1>
          </div>
          <p className="text-center text-gray-600 text-lg">
            Intelligent document processing with graph and vector search
          </p>
        </header>

        {/* Upload Section */}
        <Upload key={uploadKey} onUploadSuccess={handleUploadSuccess} />

        {/* Two Column Layout: Documents List + Chat */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
          {/* Documents List - Left Sidebar */}
          <div className="lg:col-span-1">
            <DocumentList 
              key={documentsKey}
              onSelectDocument={handleSelectDocument}
            />
          </div>

          {/* Chat Section - Main Content */}
          <div className="lg:col-span-2">
            <Chat selectedCollection={selectedCollection} />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-gray-500 text-sm">
          <p>Powered by Neo4j, ChromaDB, and Ollama • Hybrid RAG Architecture</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
