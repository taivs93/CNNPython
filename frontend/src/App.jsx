// frontend/src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import MnistPage from './pages/MnistPage';
import ShapesPage from './pages/ShapesPage';

const App = () => {
    return (
        <Router>
            <Routes>
                <Route path='/' element={<Home />} />
                <Route path='/mnist' element={<MnistPage />} />
                <Route path='/shapes' element={<ShapesPage />} />
            </Routes>
        </Router>
    );
};

export default App;
