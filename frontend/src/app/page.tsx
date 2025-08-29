'use client';

import { useState } from 'react';

// Define types for our API response data
interface TopWord {
  word: string;
  score: number;
}

interface SummaryResult {
  summary: string;
  topWords: TopWord[];
}

export default function Home() {
  const [code, setCode] = useState<string>(`def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)`);
  const [result, setResult] = useState<SummaryResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSummarize = async () => {
    if (!code) {
      setError("Please enter some code to summarize.");
      return;
    }
    
    // Reset state and start loading
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code }),
      });

      if (!response.ok) {
        // Handle HTTP errors like 500
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const data: SummaryResult = await response.json();
      setResult(data);

    } catch (err) {
      console.error('Fetch error:', err);
      setError(err instanceof Error ? err.message : "An unknown error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main style={{ fontFamily: 'Arial', maxWidth: '800px', margin: 'auto', padding: '40px' }}>
      <h1 style={{ textAlign: 'center' }}>Code Summarizer & Counterfactual Generator</h1>
      
      <div style={{ marginBottom: '20px' }}>
        <label htmlFor="codeInput" style={{ display: 'block', marginBottom: '10px', fontWeight: 'bold' }}>
          Enter your code snippet:
        </label>
        <textarea
          id="codeInput"
          value={code}
          onChange={(e) => setCode(e.target.value)}
          placeholder="def your_function(): ..."
          style={{ width: '100%', minHeight: '200px', fontFamily: 'monospace', fontSize: '14px', padding: '10px', border: '1px solid #ccc', borderRadius: '5px' }}
        />
      </div>

      <div>
        <button 
          onClick={handleSummarize}
          disabled={isLoading}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: isLoading ? '#ccc' : '#4CAF50', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          {isLoading ? 'Analyzing...' : 'Summarize Code'}
        </button>
      </div>
      
      {error && (
        <div style={{ marginTop: '20px', backgroundColor: '#ffebee', color: '#c62828', padding: '15px', borderRadius: '5px' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: '30px', backgroundColor: '#f9f9f9', padding: '20px', borderRadius: '8px', border: '1px solid #e0e0e0' }}>
          <h2 style={{ marginBottom: '15px' }}>Results</h2>
          <div style={{ backgroundColor: '#e8f5e9', padding: '15px', borderRadius: '5px', marginBottom: '15px' }}>
            <strong>Summary:</strong> {result.summary}
          </div>
          <div>
            <strong>Top Important Words:</strong>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '10px' }}>
              {result.topWords.map((item) => (
                <div key={item.word} style={{ backgroundColor: '#e3f2fd', padding: '8px 12px', borderRadius: '5px', textAlign: 'center' }}>
                  <div style={{ fontWeight: 'bold', color: '#1976d2' }}>{item.word}</div>
                  <div style={{ fontSize: '12px', color: '#666' }}>{item.score.toFixed(4)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}