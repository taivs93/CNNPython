// frontend/src/pages/ShapesPage.jsx
import React, { useState } from 'react';
import CanvasDrawing from '../components/CanvasDrawing';
import Toolbar from '../components/Toolbar';
import { predictShapes } from '../api';

const ShapesPage = () => {
    const [imageData, setImageData] = useState(null);
    const [prediction, setPrediction] = useState(null);

    const handleSubmit = async () => {
        if(!imageData) return;
        const res = await predictShapes(imageData);
        setPrediction(res);
    };

    const handleClear = () => {
        setImageData(null);
        setPrediction(null);
    };

    return (
        <div>
            <h2>Draw a Shape (Circle/Rectangle)</h2>
            <CanvasDrawing onChange={setImageData} />
            <Toolbar onSubmit={handleSubmit} onClear={handleClear} />
            {prediction && <div>Predicted Shape: {prediction.prediction}</div>}
        </div>
    );
};

export default ShapesPage;
