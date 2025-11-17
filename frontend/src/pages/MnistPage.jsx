// frontend/src/pages/MnistPage.jsx
import React, { useState } from 'react';
import CanvasDrawing from '../components/CanvasDrawing';
import Toolbar from '../components/Toolbar';
import { predictMnist } from '../api';

const MnistPage = () => {
    const [imageData, setImageData] = useState(null);
    const [prediction, setPrediction] = useState(null);

    const handleSubmit = async () => {
        if(!imageData) return;
        const res = await predictMnist(imageData);
        setPrediction(res);
    };

    const handleClear = () => {
        setImageData(null);
        setPrediction(null);
    };

    return (
        <div>
            <h2>Draw a Digit (0-9)</h2>
            <CanvasDrawing onChange={setImageData} />
            <Toolbar onSubmit={handleSubmit} onClear={handleClear} />
            {prediction && <div>Predicted Digit: {prediction.prediction}</div>}
        </div>
    );
};

export default MnistPage;
