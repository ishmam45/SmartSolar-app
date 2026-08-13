import React, { useState } from 'react';
import { motion } from 'framer-motion';

const Login = ({ onLogin }) => {
  const [isRegistering, setIsRegistering] = useState(false);
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('admin123');
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [error, setError] = useState('');

  // Mock database
  const [users, setUsers] = useState([{ user: 'admin', pass: 'admin123' }]);

  const handleLogin = (e) => {
    e.preventDefault();
    const foundUser = users.find(u => u.user === username && u.pass === password);
    if (foundUser) {
      onLogin(username);
    } else {
      setError('Invalid credentials. Access denied.');
    }
  };

  const handleRegister = (e) => {
    e.preventDefault();
    if (!newUsername || !newPassword) {
      setError('System requires all fields.');
      return;
    }
    
    if (users.find(u => u.user === newUsername)) {
      setError('User identity already exists.');
      return;
    }

    setUsers([...users, { user: newUsername, pass: newPassword }]);
    setIsRegistering(false);
    setUsername(newUsername);
    setPassword(newPassword);
    setError('Initialization complete! You can now authenticate.');
    setTimeout(() => setError(''), 3000);
  };

  return (
    <div className="login-page">
      <motion.div 
        className="login-card glass-panel"
        initial={{ opacity: 0, y: 30, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      >
        <h2>{isRegistering ? 'INITIALIZE' : 'SMARTSOLAR'}</h2>
        <p>{isRegistering ? 'Register new node identity' : 'Authenticate to access neural predictors'}</p>
        
        {error && <motion.div initial={{opacity:0}} animate={{opacity:1}} className="error-msg">{error}</motion.div>}

        {!isRegistering ? (
          <form onSubmit={handleLogin}>
            <div className="input-group">
              <label>IDENTITY</label>
              <input 
                type="text" 
                value={username} 
                onChange={(e) => setUsername(e.target.value)} 
                placeholder="Enter admin ID" 
              />
            </div>
            <div className="input-group">
              <label>PASSPHRASE</label>
              <input 
                type="password" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)} 
                placeholder="Enter passphrase" 
              />
            </div>
            <motion.button 
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              type="submit" 
              className="login-btn"
            >
              INITIATE UPLINK
            </motion.button>
          </form>
        ) : (
          <form onSubmit={handleRegister}>
            <div className="input-group">
              <label>NEW IDENTITY</label>
              <input 
                type="text" 
                value={newUsername} 
                onChange={(e) => setNewUsername(e.target.value)} 
                placeholder="Choose identifier" 
              />
            </div>
            <div className="input-group">
              <label>NEW PASSPHRASE</label>
              <input 
                type="password" 
                value={newPassword} 
                onChange={(e) => setNewPassword(e.target.value)} 
                placeholder="Set secure passphrase" 
              />
            </div>
            <motion.button 
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              type="submit" 
              className="login-btn"
            >
              REGISTER NODE
            </motion.button>
          </form>
        )}

        <div className="auth-switch">
          {isRegistering ? (
            <>Existing identity verified? <span onClick={() => setIsRegistering(false)}>Establish Uplink</span></>
          ) : (
            <>Unregistered node? <span onClick={() => setIsRegistering(true)}>Request Access</span></>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default Login;
