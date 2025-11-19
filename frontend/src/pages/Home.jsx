// frontend/src/pages/Home.jsx
import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
    return (
        <div className='card'>
            <h1 className='title'>Welcome to Drawing Recognition</h1>
            <p className='muted'>Select a mode to start drawing and get predictions.</p>
            <div className='actions'>
                <Link className='btn primary' to='/mnist'>Draw Digits</Link>
                <Link className='btn' to='/shapes'>Draw Shapes</Link>
            </div>
        </div>
    );
};

export default Home;
