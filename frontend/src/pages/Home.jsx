// frontend/src/pages/Home.jsx
import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
    return (
        <div>
            <h1>Welcome to Drawing Recognition App</h1>
            <ul>
                <li><Link to='/mnist'>Draw Digits</Link></li>
                <li><Link to='/shapes'>Draw Shapes</Link></li>
            </ul>
        </div>
    );
};

export default Home;
