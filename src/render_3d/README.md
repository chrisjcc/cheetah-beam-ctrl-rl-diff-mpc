# ARES 3D Visualization

This repository contains a Python-based simulation control system and a JavaScript-based 3D visualization application for a particle beam lattice. Follow the instructions below to set up the environment and dependencies.

## Prerequisites

Ensure you have the following installed on your system:
- [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- [Node.js & npm](https://nodejs.org/en/download/)

## Setting up the Environment

### 1. Create and Activate the Conda Environment
```bash
conda create --name rlaa2025 python=3.11 -y
conda activate rlaa2025
```

### 2. Install Python Dependencies
```bash
pip install -r beam_3d_visualizer/beam_server/requirements.txt
python -m pip install git+https://github.com/ocelot-collab/ocelot.git@v22.12.0
pip install git+https://github.com/chrisjcc/cheetah.git@feature/3d_lattice_viewer
```

### 3. Set Up JavaScript Dependencies
Navigate to the JavaScript project directory (e.g., `beam_3d_visualizer/`) and install the necessary dependencies:
```bash
cd beam_3d_visualizer
npm install
npm audit fix
```

### 4. Quick view of 3D Lattice
To start the development server, run:
```bash
npm run dev
```
Then, open http://localhost:5173/asdf/ in your preferred web browser. To stop the application, return to the terminal running the server and press Ctrl + D.

### 5. Running an Example: Particle Beam Simulation
To run an example of the particle beam simulation. Navigate to the notebooks directory:
```
cd notebooks
```

Launch Jupyter Notebook:
```
jupyter notebook
```

Open `beam_control_3d_render.ipynb` in the Jupyter Notebook interface. Run all cells in the notebook to execute the simulation. The notebook will interact with the 3D visualization application, rendering the particle beam lattice in real time.

Note: To stop Jupyter Notebook, return to the terminal and press Ctrl + C.


## Additional Notes
- If encountering issues with dependencies, try reinstalling them or using a clean environment.
- For further troubleshooting, check the respective documentation of `npm`, `pip`, or `conda`.

## License
[Specify your project's license here]

## Contact
For issues or contributions, feel free to open a GitHub issue or reach out to the maintainers.
