import React, { useRef, useEffect, forwardRef, useImperativeHandle } from 'react'
import './CanvasDrawing.css'

const CanvasDrawing = forwardRef((props, ref) => {
  const canvasRef = useRef(null)
  const isDrawing = useRef(false)

  useImperativeHandle(ref, () => ({
    getCanvas: () => canvasRef.current,
    clear: () => {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      ctx.fillStyle = 'white'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
  }))

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    // Set canvas size
    const rect = canvas.parentElement.getBoundingClientRect()
    canvas.width = rect.width - 40
    canvas.height = rect.width - 40 // Square canvas
    
    // Fill white background
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }, [])

  const startDrawing = (e) => {
    isDrawing.current = true
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const ctx = canvas.getContext('2d')
    
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    ctx.beginPath()
    ctx.moveTo(x, y)
  }

  const draw = (e) => {
    if (!isDrawing.current) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const ctx = canvas.getContext('2d')
    
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    ctx.lineTo(x, y)
    ctx.strokeStyle = 'black'
    ctx.lineWidth = 4
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.stroke()
  }

  const stopDrawing = () => {
    isDrawing.current = false
  }

  // Touch events for mobile
  const startTouchDrawing = (e) => {
    isDrawing.current = true
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const ctx = canvas.getContext('2d')
    
    const touch = e.touches[0]
    const x = touch.clientX - rect.left
    const y = touch.clientY - rect.top

    ctx.beginPath()
    ctx.moveTo(x, y)
  }

  const touchDraw = (e) => {
    if (!isDrawing.current) return
    e.preventDefault()

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const ctx = canvas.getContext('2d')
    
    const touch = e.touches[0]
    const x = touch.clientX - rect.left
    const y = touch.clientY - rect.top

    ctx.lineTo(x, y)
    ctx.strokeStyle = 'black'
    ctx.lineWidth = 4
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.stroke()
  }

  const stopTouchDrawing = () => {
    isDrawing.current = false
  }

  return (
    <canvas
      ref={canvasRef}
      className="drawing-canvas"
      onMouseDown={startDrawing}
      onMouseMove={draw}
      onMouseUp={stopDrawing}
      onMouseLeave={stopDrawing}
      onTouchStart={startTouchDrawing}
      onTouchMove={touchDraw}
      onTouchEnd={stopTouchDrawing}
    />
  )
})

CanvasDrawing.displayName = 'CanvasDrawing'

export default CanvasDrawing
