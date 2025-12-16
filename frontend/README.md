# HybridRAG Frontend

Modern React UI for the HybridRAG Knowledge Engine with file upload and interactive chat.

## Features

- 📤 **File Upload**: Drag-and-drop PDF upload with real-time processing feedback
- 💬 **Interactive Chat**: Conversation interface with hybrid RAG capabilities
- 📊 **Source Display**: Expandable panels showing both vector and graph sources
- 🔗 **Provenance Tracking**: View graph nodes and document chunks used in answers
- 🎨 **Modern UI**: Beautiful Tailwind CSS styling with animations
- ⚡ **Real-time Updates**: Live feedback on uploads and queries

## Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

## Installation

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```

3. **Open your browser:**
   Navigate to `http://localhost:5173`

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Upload.jsx       # File upload component
│   │   ├── Chat.jsx         # Chat interface with sources
│   │   └── SourceList.jsx   # (Legacy) Source list component
│   ├── App.jsx              # Main application component
│   ├── index.jsx            # Application entry point
│   └── index.css            # Tailwind CSS and custom styles
├── index.html               # HTML template
├── vite.config.js           # Vite configuration with API proxy
├── tailwind.config.js       # Tailwind CSS configuration
├── postcss.config.js        # PostCSS configuration
└── package.json             # Dependencies and scripts
```

## Component Overview

### Upload Component (`Upload.jsx`)

Handles PDF file uploads to the backend.

**Features:**
- File selection with visual feedback
- Upload progress indication
- Success summary showing:
  - Document ID
  - Number of chunks created
  - Number of entities extracted
  - Number of relationships found
- Error handling with user-friendly messages

**API Endpoint:** `POST /api/ingest`

### Chat Component (`Chat.jsx`)

Interactive chat interface for querying the knowledge base.

**Features:**
- Message history with user/assistant roles
- Real-time typing indicators
- Expandable source panels showing:
  - Vector search results (with document snippets)
  - Graph search results (with entity information)
  - Provenance data (graph node IDs used)
- Auto-scroll to latest message
- Error handling for failed queries

**API Endpoint:** `POST /api/query`

**Request Format:**
```json
{
  "question": "What is Python?",
  "top_k": 3,
  "use_hybrid": true
}
```

**Response Format:**
```json
{
  "answer": "Python is...",
  "sources": [
    {
      "type": "vector" | "graph",
      "content": "...",
      "metadata": { "doc_id": "...", ... }
    }
  ],
  "graph_context": "=== Knowledge Graph Context ===\n...",
  "provenance": {
    "chunk_ids": ["..."],
    "node_ids": ["Python|Language", "..."],
    "vector_doc_ids": ["..."]
  }
}
```

## API Proxy Configuration

The Vite dev server proxies API requests to the backend:

```javascript
// vite.config.js
proxy: {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  },
}
```

This means requests to `/api/ingest` → `http://localhost:8000/api/ingest`

## Customization

### Styling

The UI uses Tailwind CSS. Customize colors and styles in:
- `tailwind.config.js` - Theme configuration
- `src/index.css` - Custom CSS and animations

### API Configuration

To change the backend URL:
1. Edit `vite.config.js` proxy target
2. Or set environment variable: `VITE_API_URL`

## Build for Production

```bash
npm run build
```

This creates a `dist/` folder with optimized static files.

### Serve Production Build

```bash
npm run preview
```

## Troubleshooting

### API Connection Issues

**Problem:** Requests to `/api/*` fail with network errors

**Solution:**
1. Ensure backend is running on `http://localhost:8000`
2. Check backend logs for CORS errors
3. Verify proxy configuration in `vite.config.js`

### Tailwind Styles Not Loading

**Problem:** UI appears unstyled

**Solution:**
1. Ensure `npm install` completed successfully
2. Check that `tailwind.config.js` and `postcss.config.js` exist
3. Restart dev server: `npm run dev`

### Upload Fails

**Problem:** File upload returns 500 error

**Solution:**
1. Check backend logs for detailed error
2. Verify file is a valid PDF
3. Ensure backend has access to Neo4j and ChromaDB
4. Check OPENAI_API_KEY is set in backend environment

## Development Tips

### Hot Reload

Vite provides instant hot module replacement (HMR). Changes to components will reflect immediately without full page reload.

### Component Development

Use React DevTools browser extension to inspect component state and props.

### API Testing

Use browser DevTools Network tab to inspect API requests and responses.

## Scripts

- `npm run dev` - Start development server (with HMR)
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally

## Dependencies

### Core
- **React 18** - UI library
- **React DOM** - React renderer
- **Vite** - Build tool and dev server

### Styling
- **Tailwind CSS** - Utility-first CSS framework
- **PostCSS** - CSS processing
- **Autoprefixer** - CSS vendor prefixing

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Performance

- **Code Splitting**: Automatic via Vite
- **Tree Shaking**: Dead code elimination
- **Minification**: Production builds are minified
- **Asset Optimization**: Images and fonts optimized

## Accessibility

- Semantic HTML elements
- ARIA labels where appropriate
- Keyboard navigation support
- Focus management in modals/dropdowns

## License

MIT
