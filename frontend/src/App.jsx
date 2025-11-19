// frontend/src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import MnistPage from './pages/MnistPage';
import ShapesPage from './pages/ShapesPage';

const App = () => {
    return (
        <Router>
            <div className='app'>
                <header className='header'>
                    <div className='brand'>Drawing Recognition</div>
                    <nav className='nav'>
                        <Link to='/'>Home</Link>
                        <Link to='/mnist'>Digits</Link>
                        <Link to='/shapes'>Shapes</Link>
                    </nav>
                </header>
                <main className='container'>
                    <Routes>
                        <Route path='/' element={<Home />} />
                        <Route path='/mnist' element={<MnistPage />} />
                        <Route path='/shapes' element={<ShapesPage />} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
};

export default App;
