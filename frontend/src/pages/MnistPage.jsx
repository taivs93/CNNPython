import React, { useState, useRef } from 'react'
import './PredictionPage.css'
import CanvasDrawing from '../components/CanvasDrawing'
import ResultDisplay from '../components/ResultDisplay'
import { predictMnist } from '../api'

export default function MnistPage({ onBack }) {
  const canvasRef = useRef(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePredict = async () => {
    if (!canvasRef.current) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const canvas = canvasRef.current.getCanvas()
      canvas.toBlob(async (blob) => {
        const file = new File([blob], 'drawing.png', { type: 'image/png' })
        const response = await predictMnist(file)
        
        if (response.success) {
          setResult({
            ...response,
            type: 'mnist'
          })
        } else {
          setError(response.error || 'Dự đoán thất bại')
        }
      })
    } catch (err) {
      setError(err.message || 'Lỗi server')
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    if (canvasRef.current) {
      canvasRef.current.clear()
    }
    setResult(null)
    setError(null)
  }

  return (
    <div className="prediction-page">
      <div className="page-header">
        <button className="back-btn" onClick={onBack}>← Quay lại</button>
        <h1>🔢 Nhận diện chữ số</h1>
      </div>

      <div className="page-content">
        <div className="canvas-section">
          <div className="canvas-box">
            <CanvasDrawing ref={canvasRef} />
          </div>

          <div className="canvas-controls">
            <button 
              className="btn-clear"
              onClick={handleClear}
              disabled={loading}
            >
              🗑️ Xóa
            </button>
            <button 
              className="btn-predict"
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? '⏳ Đang dự đoán...' : '🎯 Dự đoán'}
            </button>
          </div>
        </div>

        <div className="result-section">
          {error && (
            <div className="error-box">
              <p>❌ {error}</p>
            </div>
          )}
          {result && (
            <ResultDisplay result={result} type="mnist" />
          )}
          {!result && !error && (
            <div className="placeholder">
              <p>📊 Vẽ một chữ số (0-9) trên canvas</p>
              <p>rồi nhấn "Dự đoán"</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
