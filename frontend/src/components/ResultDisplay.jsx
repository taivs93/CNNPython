import React from 'react';
import './ResultDisplay.css';

export default function ResultDisplay({ result, type }) {
  if (!result) return null;

  const { success, prediction, confidence_percent, error } = result;

  if (!success) {
    return (
      <div className="result-container error">
        <h3>❌ Lỗi dự đoán</h3>
        <p>{error}</p>
      </div>
    );
  }

  const displayText = type === 'mnist' ? `Chữ số: ${prediction}` : `Hình dạng: ${prediction}`;

  return (
    <div className="result-container success">
      <div className="result-main">
        <h2>{displayText}</h2>
        <div className="confidence-wrapper">
          <div className="confidence-bar">
            <div 
              className="confidence-fill" 
              style={{ width: `${confidence_percent}%` }}
            >
            </div>
          </div>
          <p className="confidence-text">
            Độ tin cậy: <strong>{confidence_percent}%</strong>
          </p>
        </div>
      </div>
    </div>
  );
}
