// frontend/src/components/Toolbar.jsx
import React from 'react';

const Toolbar = ({ onSubmit, onClear }) => {
    return (
        <div className='toolbar'>
            <button className='btn primary' onClick={onSubmit}>Predict</button>
            <button className='btn' onClick={onClear}>Clear</button>
        </div>
    );
};

export default Toolbar;