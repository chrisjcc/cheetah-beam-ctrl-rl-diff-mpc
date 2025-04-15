import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Since __dirname is not available in ESM, derive it from import.meta.url
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

// Default Vite hostname and port or env variable
const FRONTEND_PORT = process.env.VITE_FRONTEND_PORT || 5173;
const PROXY_HOST_NAME = process.env.VITE_PROXY_HOST_NAME || "127.0.0.1";

// Serve static files from the 'dist' directory
const distPath = join(__dirname, 'dist');

//app.use(express.static(join(__dirname, 'beam_3d_visualizer/dist')));

// Serve static files from the Vite build output (dist folder)
app.use(express.static(distPath, {
  setHeaders: (res, filePath) => {
    // Ensure JavaScript files are served with the correct MIME type
    if (filePath.endsWith('.js')) {
      res.setHeader('Content-Type', 'application/javascript');
    }
    // Ensure CSS files are served with the correct MIME type
    if (filePath.endsWith('.css')) {
      res.setHeader('Content-Type', 'text/css');
    }
    // Ensure .gltf files are served with the correct MIME type
    if (filePath.endsWith('.gltf')) {
      res.setHeader('Content-Type', 'application/json');
    }
    // Ensure .bin files are served with the correct MIME type
    if (filePath.endsWith('.bin')) {
      res.setHeader('Content-Type', 'application/octet-stream');
    }
  }
}));

// Handle all routes by serving index.html (for client-side routing)
app.get('*', (req, res) => {
  res.sendFile(join(distPath, 'index.html'));
});

app.listen(FRONTEND_PORT, () => {
  console.log(`Production Server running at http://${PROXY_HOST_NAME}:${FRONTEND_PORT}`);
});
