// frontend/src/components/Toolbar.jsx
import React from 'react';

const Toolbar = ({ onSubmit, onClear }) => {
    return (
        <div style={{margin:'10px 0'}}>
            <button onClick={onSubmit}>Predict</button>
            <button onClick={onClear} style={{marginLeft:'10px'}}>Clear</button>
        </div>
    );
};

export default Toolbar;