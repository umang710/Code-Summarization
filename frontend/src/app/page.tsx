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

interface CounterfactualResult {
  label: string;
  code: string;
  summary: string;
  topWords: TopWord[];
}

export default function Home() {
  const [code, setCode] = useState<string>(`def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)`);
  const [summaryResult, setSummaryResult] = useState<SummaryResult | null>(null);
  const [counterfactuals, setCounterfactuals] = useState<CounterfactualResult[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isCfLoading, setIsCfLoading] = useState<boolean>(false);

  const handleSummarize = async () => {
    if (!code) {
      setError("Please enter some code to summarize.");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setSummaryResult(null);
    setCounterfactuals([]); // Clear old counterfactuals

    try {
      const response = await fetch('/api/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      if (!response.ok) throw new Error(`Server responded with status: ${response.status}`);
      const data: SummaryResult = await response.json();
      setSummaryResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateCounterfactuals = async () => {
    if (!code) {
      setError("Please summarize the original code first.");
      return;
    }

    setIsCfLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/counterfactuals', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code }),
      });
      if (!response.ok) throw new Error(`Server responded with status: ${response.status}`);
      const data = await response.json();
      setCounterfactuals(data.counterfactuals);
    } catch (err) {
        setError(err instanceof Error ? err.message : "An unknown error occurred.");
    } finally {
        setIsCfLoading(false);
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

      <div style={{ display: 'flex', gap: '10px' }}>
        <button 
          onClick={handleSummarize}
          disabled={isLoading}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: isLoading ? '#ccc' : '#4CAF50', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          {isLoading ? 'Analyzing...' : 'Summarize Code'}
        </button>
        <button 
          onClick={handleGenerateCounterfactuals}
          disabled={!summaryResult || isCfLoading}
          style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer', backgroundColor: (!summaryResult || isCfLoading) ? '#ccc' : '#2196F3', color: 'white', border: 'none', borderRadius: '5px' }}
        >
          {isCfLoading ? 'Generating...' : 'Generate Counterfactuals'}
        </button>
      </div>
      
      {error && (
        <div style={{ marginTop: '20px', backgroundColor: '#ffebee', color: '#c62828', padding: '15px', borderRadius: '5px' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {summaryResult && (
        <div style={{ marginTop: '30px', backgroundColor: '#f9f9f9', padding: '20px', borderRadius: '8px', border: '1px solid #e0e0e0' }}>
          <h2>Original Code Results</h2>
          <div style={{ backgroundColor: '#e8f5e9', padding: '15px', borderRadius: '5px', marginBottom: '15px' }}>
            <strong>Summary:</strong> {summaryResult.summary}
          </div>
          <div>
            <strong>Top Important Words:</strong>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '10px' }}>
              {summaryResult.topWords.map((item) => (
                <div key={item.word} style={{ backgroundColor: '#e3f2fd', padding: '8px 12px', borderRadius: '5px' }}>
                  <div style={{ fontWeight: 'bold', color: '#1976d2' }}>{item.word}</div>
                  <div style={{ fontSize: '12px', color: '#666' }}>{item.score.toFixed(4)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {counterfactuals.length > 0 && (
        <div style={{ marginTop: '30px' }}>
          <h2>Counterfactual Variations</h2>
          {counterfactuals.map((cf) => (
            <div key={cf.label} style={{ marginTop: '20px', backgroundColor: '#f9f9f9', padding: '20px', borderRadius: '8px', border: '1px solid #e0e0e0' }}>
              <h3 style={{ textTransform: 'capitalize', color: '#1976d2' }}>{cf.label.replace(/_/g, ' ')}</h3>
              <pre style={{ backgroundColor: '#2d2d2d', color: '#f8f8f2', padding: '15px', borderRadius: '5px', overflowX: 'auto' }}>
                <code>{cf.code}</code>
              </pre>
              <div style={{ backgroundColor: '#e8f5e9', padding: '15px', borderRadius: '5px', margin: '15px 0' }}>
                <strong>Summary:</strong> {cf.summary}
              </div>
              <strong>Top Important Words:</strong>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '10px' }}>
                {cf.topWords.map((item) => (
                  <div key={item.word} style={{ backgroundColor: '#e3f2fd', padding: '8px 12px', borderRadius: '5px' }}>
                    <div style={{ fontWeight: 'bold', color: '#1976d2' }}>{item.word}</div>
                    <div style={{ fontSize: '12px', color: '#666' }}>{item.score.toFixed(4)}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}

//re-Deploy