import { useState } from 'react';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    // Add user message
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // TODO: Implement query API call
      // - POST to /api/query/ with question
      // - Handle response with answer and sources
      // - Add assistant message to chat
      
      // Simulated response
      setTimeout(() => {
        const assistantMessage = {
          role: 'assistant',
          content: 'Response will be implemented here.',
          sources: []
        };
        setMessages(prev => [...prev, assistantMessage]);
        setLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Error querying:', error);
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <h2>Ask Questions</h2>
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'You' : 'Assistant'}:</strong>
            <p>{msg.content}</p>
          </div>
        ))}
        {loading && <div className="message assistant">Thinking...</div>}
      </div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

export default Chat;

