import React from 'react'
import './Home.css'

export default function Home({ onSelectMode }) {
  return (
    <div className="home-container">
      <div className="home-content">
        <h1 className="home-title">🎨 CNN Drawing Predictor</h1>
        <p className="home-subtitle">Chọn loại dự đoán</p>
        
        <div className="mode-buttons">
          <button 
            className="mode-btn mnist-btn"
            onClick={() => onSelectMode('mnist')}
          >
            <span className="btn-icon">🔢</span>
            <span className="btn-text">Nhận diện chữ số</span>
            <span className="btn-desc">Vẽ chữ số từ 0-9</span>
          </button>
          
          <button 
            className="mode-btn shapes-btn"
            onClick={() => onSelectMode('shapes')}
          >
            <span className="btn-icon">⭕</span>
            <span className="btn-text">Nhận diện hình dạng</span>
            <span className="btn-desc">Vẽ hình tròn hoặc hình chữ nhật</span>
          </button>
        </div>

        <div className="home-info">
          <p>💡 Vẽ rõ ràng trên canvas để có kết quả tốt nhất</p>
        </div>
      </div>
    </div>
  )
}
