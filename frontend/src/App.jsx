import { useState } from 'react';
import Upload from './components/Upload';
import Chat from './components/Chat';
import SourceList from './components/SourceList';
import './App.css';

function App() {
  const [sources, setSources] = useState([]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>HybridRAG Knowledge Engine</h1>
      </header>
      <main>
        <Upload onUploadSuccess={() => {
          // TODO: Fetch updated sources list after upload
        }} />
        <div className="main-content">
          <Chat />
          <SourceList sources={sources} />
        </div>
      </main>
    </div>
  );
}

export default App;

