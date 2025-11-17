// frontend/src/components/SampleImages.jsx
import React from 'react';

const SampleImages = ({ images, onClick }) => {
    return (
        <div style={{display:'flex', flexWrap:'wrap'}}>
            {images.map((img, idx) => (
                <img key={idx} src={img} alt={`sample-${idx}`} style={{width:'60px', margin:'5px', cursor:'pointer'}} onClick={() => onClick(img)} />
            ))}
        </div>
    );
};

export default SampleImages;
