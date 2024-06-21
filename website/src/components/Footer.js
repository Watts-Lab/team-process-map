import React from 'react';
import './Footer.css';
// import { Button } from './Button';
// import { Link } from 'react-router-dom';

function Footer() {
    return (
        <div className='footer-container'>
            <div className='logo-links'>
                <a href='https://css.seas.upenn.edu/'>
                    <img src='csslab_logo.png' alt='csslabLogo' />
                </a>
                <a href='https://ai-analytics.wharton.upenn.edu/'>
                    <img src='wharton_logo_white.png' alt='whartonLogo' />
                </a>
            </div>
        </div>
    );
}

export default Footer;