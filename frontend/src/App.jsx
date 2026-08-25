import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const CATEGORIES = [
  { key: 'brushing', label: 'Brushing', emoji: '🪮', color: 'var(--series-1)' },
  { key: 'isolation', label: 'Isolation', emoji: '😿', color: 'var(--series-2)' },
  { key: 'waiting_for_food', label: 'Waiting for food', emoji: '🍽️', color: 'var(--series-3)' },
]

function App() {
  const [file, setFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [result, setResult] = useState(null)
  const [errorMsg, setErrorMsg] = useState('')

  function handleFileChange(event) {
    const selected = event.target.files?.[0]
    if (!selected) return

    setFile(selected)
    setAudioUrl(URL.createObjectURL(selected))
    setStatus('idle')
    setResult(null)
    setErrorMsg('')
  }

  async function handleAnalyze() {
    if (!file) return
    setStatus('loading')
    setErrorMsg('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed (${response.status})`)
      }

      const data = await response.json()
      setResult(data)
      setStatus('success')
    } catch (err) {
      setErrorMsg(err.message || 'Something went wrong')
      setStatus('error')
    }
  }

  const winningKey = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])[0][0]
    : null
  const winningCategory = CATEGORIES.find((c) => c.key === winningKey)

  return (
    <div className="app">
      <header className="app-header">
        <h1>🐱 Cat Mood Classifier</h1>
        <p>Upload a meow clip and find out what your cat is trying to say.</p>
      </header>

      <div className="card">
        <label className="dropzone">
          <input type="file" accept=".wav" onChange={handleFileChange} />
          {file ? (
            <span className="filename">{file.name}</span>
          ) : (
            <span>Click to upload a .wav clip</span>
          )}
        </label>

        {audioUrl && (
          <audio className="audio-preview" controls src={audioUrl}></audio>
        )}

        <button
          className="analyze-button"
          disabled={!file || status === 'loading'}
          onClick={handleAnalyze}
        >
          {status === 'loading' ? 'Analyzing...' : 'Analyze'}
        </button>

        {status === 'error' && <p className="error-text">{errorMsg}</p>}
      </div>

      {status === 'success' && result && (
        <div className="card">
          <div className="result-heading">
            <span
              className="dot"
              style={{ background: winningCategory?.color }}
            ></span>
            <div>
              <h2>
                {winningCategory?.emoji} {winningCategory?.label}
              </h2>
              <p className="sub">This is our best guess based on the clip.</p>
            </div>
          </div>

          {CATEGORIES.map((category) => {
            const prob = result.probabilities[category.key] ?? 0
            return (
              <div
                key={category.key}
                className={`bar-row ${category.key === winningKey ? 'is-winner' : ''}`}
              >
                <span className="bar-label">
                  {category.emoji} {category.label}
                </span>
                <span className="bar-track">
                  <span
                    className="bar-fill"
                    style={{
                      width: `${prob * 100}%`,
                      background: category.color,
                    }}
                  ></span>
                </span>
                <span className="bar-value">{Math.round(prob * 100)}%</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default App
