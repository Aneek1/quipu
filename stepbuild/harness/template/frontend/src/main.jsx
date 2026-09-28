import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

// Compile every component even before App.jsx imports it: step 4 writes the
// components and runs `npm run build`, but nothing imports them until step 5.
import.meta.glob('./components/*.jsx', { eager: true })

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
