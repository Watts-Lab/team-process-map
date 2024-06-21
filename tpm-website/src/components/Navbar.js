import React, {useState} from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';
// import { Button } from './Button';
// import { slide as Menu } from 'react-burger-menu';
import { FaBars, FaTimes } from 'react-icons/fa';

function Navbar() {
    const [click, setClick] = useState(false);
    // const[button, setButton] = useState(true);

    const handleClick = () => setClick(!click);
    const closeMobileMenu = () => setClick(false);

    // const showButton = () => {
    //     if(window.innerWidth <= 960) {
    //         setButton(false)
    //     } else {
    //         setButton(true);
    //     }
    // };

    // window.addEventListener('resize', showButton);

    return (
        <>
            <nav className="navbar">
                <div className="navbar-container">
                    <Link to="/" className="navbar-logo">
                        Team Process Mapping
                    </Link>
                    <div className='menu-icon' onClick={handleClick}>
                        {click ? <FaTimes /> : <FaBars />}
                    </div>
                    <ul className={click ? 'nav-menu active' : 'nav-menu'}>
                        <li className='nav-item'>
                            <Link to='/' className='nav-links' onClick={closeMobileMenu}>
                                Home
                            </Link>
                        </li>
                        <li className='nav-item'>
                            <Link to='/Research' className='nav-links' onClick={closeMobileMenu}>
                                Research
                            </Link>
                        </li>
                        <li className='nav-item'>
                            <Link to='/HowItWorks' className='nav-links' onClick={closeMobileMenu}>
                                How It Works
                            </Link>
                        </li>
                        <li className='nav-item'>
                            <Link to='/Team' className='nav-links' onClick={closeMobileMenu}>
                                Team
                            </Link>
                        </li>
                    </ul>
                    {/* {button && <Button>SIGN UP</Button>} */}
                </div>
            </nav>
        </>
  )
}

export default Navbar