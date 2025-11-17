// frontend/src/components/CanvasDrawing.jsx
import React, { useRef, useState, useEffect } from 'react';

const CanvasDrawing = ({ width=280, height=280, onChange }) => {
    const canvasRef = useRef();
    const [ctx, setCtx] = useState(null);
    const [drawing, setDrawing] = useState(false);
    const [history, setHistory] = useState([]);
    const [redoStack, setRedoStack] = useState([]);

    useEffect(() => {
        const canvas = canvasRef.current;
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d');
        context.fillStyle = 'white';
        context.fillRect(0, 0, width, height);
        context.lineWidth = 8;
        context.lineCap = 'round';
        setCtx(context);
        setHistory([context.getImageData(0,0,width,height)]);
    }, []);

    const startDrawing = e => {
        setDrawing(true);
        ctx.beginPath();
        ctx.moveTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    };
    const draw = e => {
        if(!drawing) return;
        ctx.lineTo(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
        ctx.stroke();
    };
    const stopDrawing = () => {
        if(!drawing) return;
        setDrawing(false);
        setHistory(prev => [...prev, ctx.getImageData(0,0,width,height)]);
        setRedoStack([]);
        onChange && onChange(canvasRef.current.toDataURL());
    };

    const undo = () => {
        if(history.length <= 1) return;
        const newHistory = [...history];
        const last = newHistory.pop();
        setRedoStack(prev => [...prev, last]);
        setHistory(newHistory);
        ctx.putImageData(newHistory[newHistory.length-1],0,0);
        onChange && onChange(canvasRef.current.toDataURL());
    };

    const redo = () => {
        if(redoStack.length===0) return;
        const newRedo = [...redoStack];
        const imgData = newRedo.pop();
        ctx.putImageData(imgData,0,0);
        setHistory(prev => [...prev, imgData]);
        setRedoStack(newRedo);
        onChange && onChange(canvasRef.current.toDataURL());
    };

    const clear = () => {
        ctx.fillStyle='white';
        ctx.fillRect(0,0,width,height);
        setHistory([ctx.getImageData(0,0,width,height)]);
        setRedoStack([]);
        onChange && onChange(canvasRef.current.toDataURL());
    };

    return (
        <div>
            <canvas
                ref={canvasRef}
                style={{border:'1px solid black'}}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
            />
            <div style={{marginTop:'10px'}}>
                <button onClick={undo}>Undo</button>
                <button onClick={redo}>Redo</button>
                <button onClick={clear}>Clear</button>
            </div>
        </div>
    );
};

export default CanvasDrawing;
