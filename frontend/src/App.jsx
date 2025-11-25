import React, { useState } from 'react'
import './App.css'
import Home from './pages/Home'
import MnistPage from './pages/MnistPage'
import ShapesPage from './pages/ShapesPage'

export default function App() {
  const [mode, setMode] = useState(null) // null | 'mnist' | 'shapes'

  const handleBack = () => setMode(null)

  return (
    <div className="app-container">
      {mode === null && <Home onSelectMode={setMode} />}
      {mode === 'mnist' && <MnistPage onBack={handleBack} />}
      {mode === 'shapes' && <ShapesPage onBack={handleBack} />}
    </div>
  )
}