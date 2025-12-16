# Frontend Quick Start Guide

## 🚀 Getting Started

### 1. Install Dependencies

```bash
cd frontend
npm install
```

This will install:
- React and React DOM
- Vite (build tool)
- Tailwind CSS (styling)
- React Router DOM (routing)

### 2. Start Development Server

```bash
npm run dev
```

The frontend will start on **http://localhost:5173**

### 3. Ensure Backend is Running

The frontend expects the backend API to be running on **http://localhost:8000**

Start the backend:
```bash
cd ../backend
python -m uvicorn app.main:app --reload --port 8000
```

### 4. Open in Browser

Navigate to: **http://localhost:5173**

## 📋 Usage

### Uploading Documents

1. Click the file upload area or drag a PDF file
2. Click "Upload and Process"
3. Wait for processing (shows entity extraction progress)
4. View summary with:
   - Number of chunks created
   - Number of entities extracted
   - Number of relationships found

### Chatting with Knowledge Base

1. Type your question in the chat input
2. Press Enter or click Send button
3. View the AI response
4. Click "Show Sources" to see:
   - Vector search results (text snippets)
   - Graph search results (entity relationships)
   - Graph nodes used in the answer

## 🎨 UI Features

### Upload Component
- ✅ Drag-and-drop file selection
- ✅ File type validation (PDF only)
- ✅ Upload progress indicator
- ✅ Success/error notifications
- ✅ Ingestion summary display

### Chat Component
- ✅ Message history
- ✅ Real-time responses
- ✅ Expandable source panels
- ✅ Graph node provenance
- ✅ Auto-scroll to latest message
- ✅ Loading indicators

## 🔧 Configuration

### Change Backend URL

Edit `vite.config.js`:

```javascript
proxy: {
  '/api': {
    target: 'http://your-backend-url:8000',
    changeOrigin: true,
  },
}
```

### Customize Styling

Edit `tailwind.config.js` for theme customization:

```javascript
theme: {
  extend: {
    colors: {
      primary: '#your-color',
    },
  },
}
```

## 🐛 Troubleshooting

### Port Already in Use

If port 5173 is busy:
```bash
npm run dev -- --port 3000
```

### CORS Errors

Ensure backend has CORS middleware configured:
```python
# backend/app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### API 404 Errors

Check that:
1. Backend is running on port 8000
2. API endpoints match: `/api/ingest` and `/api/query`
3. Vite proxy is configured correctly

### Tailwind Not Working

Restart the dev server:
```bash
# Stop with Ctrl+C, then:
npm run dev
```

## 📦 Build for Production

```bash
npm run build
```

Output will be in `dist/` folder.

### Preview Production Build

```bash
npm run preview
```

### Deploy

The `dist/` folder can be deployed to:
- Netlify
- Vercel
- AWS S3 + CloudFront
- Any static hosting service

## 🔑 Environment Variables

Create `.env` file (optional):

```bash
VITE_API_URL=http://localhost:8000
```

Use in code:
```javascript
const API_URL = import.meta.env.VITE_API_URL || '';
```

## 📱 Responsive Design

The UI is fully responsive and works on:
- Desktop (optimal)
- Tablet
- Mobile (chat-focused layout)

## ⌨️ Keyboard Shortcuts

- **Enter** - Send chat message
- **Esc** - Clear input (when focused)
- **Tab** - Navigate between UI elements

## 🎯 Next Steps

1. Upload a PDF document
2. Wait for processing to complete
3. Ask questions about the document
4. Explore sources and graph nodes
5. Try different query types (who, what, which, depends, list, related)

## 📚 Example Queries

**Entity-focused:**
- "Who created Python?"
- "What is FastAPI?"

**Relationship-focused:**
- "What depends on Python?"
- "List all frameworks"

**General:**
- "Explain the features of [topic]"
- "How is [X] used in [Y]?"

Enjoy using HybridRAG! 🎉
