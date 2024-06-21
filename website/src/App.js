// import logo from './logo.svg';
import './App.css';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './components/pages/Home';
import Research from './components/pages/Research';
import Team from './components/pages/Team';
import HowItWorks from './components/pages/HowItWorks';

function App() {
  return (
    <>
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/Research" element={<Research />} />
          <Route path="/Team" element={<Team />} />
          <Route path="/HowItWorks" element={<HowItWorks />} />
        </Routes>
        <Footer />
      </Router>
    </>
  );
}

export default App;
