// frontend/src/components/HistorySidebar.jsx
import React from 'react';

const HistorySidebar = ({ history, onSelect }) => {
    return (
        <div style={{border:'1px solid #ccc', padding:'5px', maxHeight:'200px', overflowY:'auto'}}>
            <h4>History</h4>
            {history.map((img, idx) => (
                <img key={idx} src={img} alt={`hist-${idx}`} style={{width:'50px', margin:'2px', cursor:'pointer'}} onClick={() => onSelect(idx)} />
            ))}
        </div>
    );
};

export default HistorySidebar;