# Stage 1: Build the frontend with Node.js
FROM node:18 AS frontend-builder

# Install Node.js, npm, and supervisor
WORKDIR /app/beam_3d_visualizer
COPY beam_3d_visualizer/package*.json ./
RUN npm install
COPY beam_3d_visualizer/ ./
COPY beam_3d_visualizer/.env* ./
RUN npm run build

# Stage 2: Build the final image with Python and Node.js
FROM python:3.11-slim

# Install Node.js, npm, supervisor, and git
RUN apt-get update --fix-missing && apt-get install -y \
    nodejs \
    npm \
    supervisor \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements
COPY beam_3d_visualizer/beam_server/requirements.txt ./beam_3d_visualizer/beam_server/

# Install Python requirements and additional Git-based dependencies
RUN pip install --no-cache-dir -r beam_3d_visualizer/beam_server/requirements.txt \
    && python -m pip install git+https://github.com/ocelot-collab/ocelot.git@v22.12.0 \
    && pip install git+https://github.com/chrisjcc/cheetah.git@feature/3d_lattice_viewer

# Copy Python backend files
COPY runner.py .
COPY beam_3d_visualizer/beam_server/ ./beam_3d_visualizer/beam_server/
COPY src/ ./src/

# Copy frontend production files from Stage 1
COPY --from=frontend-builder /app/beam_3d_visualizer/dist ./beam_3d_visualizer/dist
COPY beam_3d_visualizer/server.js ./beam_3d_visualizer/
COPY beam_3d_visualizer/package*.json ./beam_3d_visualizer/
COPY beam_3d_visualizer/.env* ./beam_3d_visualizer/

WORKDIR /app/beam_3d_visualizer
RUN npm install --omit=dev
WORKDIR /app

# Copy supervisor configuration
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports for Express (frontend) and WebSocket (backend) servers
EXPOSE 5173
EXPOSE 8081

# Start supervisor to manage both processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
