import asyncio
import logging
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from dotenv import load_dotenv
from gymnasium import Wrapper

from src.render_3d.beam_server.segment_3d_builder import Segment3DBuilder
from src.render_3d.beam_server.websocket_wrapper import WebSocketWrapper

# Calculate the path to the .env file, one levels up from the script's location
script_dir = Path(__file__).resolve().parent  # Directory of the current script
env_path = script_dir.parent / ".env"  # Two levels up to beam_3d_visualizer/.env

# Load the .env file
load_dotenv(dotenv_path=env_path)

# Set logging level based on environment
debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Setup logging with conditional log level
log_level = (
    logging.DEBUG if debug_mode else logging.WARNING
)  # Set to WARNING to suppress info/debug logs
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"Loaded .env from {env_path}")
logger.info(f"NODE_ENV: {os.getenv('NODE_ENV')}")
logger.info(f"VITE_FRONTEND_PORT: {os.getenv('VITE_FRONTEND_PORT')}")

# Define constants at module level
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 5173
DEFAULT_NUM_PARTICLES = 1000
BEAM_SOURCE_COMPONENT = "AREASOLA1"
SCREEN_NAME = "AREABSCR1"
DEFAULT_SCREEN_RESOLUTION = (2448, 2040)
DEFAULT_SCREEN_PIXEL_SIZE = (3.3198e-6, 2.4469e-6)
DEFAULT_SCREEN_BINNING = 4
DEFAULT_RENDER_MODE = "human"


class BeamVisualizationWrapper(Wrapper):
    """
    A Gym wrapper that encapsulates the beam simulation logic and manages the
    initialization of the JavaScript web application for 3D visualization.
    """

    def __init__(
        self,
        env: gym.Env,
        http_host: str = DEFAULT_HTTP_HOST,
        http_port: int = DEFAULT_HTTP_PORT,
        is_export_enabled: bool = False,
        num_particles: int = DEFAULT_NUM_PARTICLES,
        render_mode: str = DEFAULT_RENDER_MODE,
    ):
        """
        Initialize the BeamVisualizationWrapper.

        Args:
            env (gym.Env): The underlying Gym environment (e.g., BeamControlEnv).
            http_host (str): Hostname for the JavaScript web application server.
            http_port (int): Port for the web application server.
            is_export_enabled (bool): Whether to enable 3D scene export.
            num_particles (int): Number of particles to simulate in the beam.
        """
        # Internally wrap the environment with WebSocketWrapper
        self.ws_env = WebSocketWrapper(env)
        super().__init__(self.ws_env)

        # Basic configuration
        self.base_path = Path(__file__).resolve().parent
        self.http_host = http_host
        self.http_port = http_port
        self.render_mode = render_mode
        self.num_particles = num_particles
        self.current_step = 0
        self.web_process = None
        self.web_thread = None
        self.data = OrderedDict()
        self.is_export_enabled = is_export_enabled
        self.screen_reading = None

        # Initialize state
        self.incoming_particle_beam = None
        self.last_action = np.zeros(5, dtype=np.float32)

        # Ensures the necessary npm dependencies are installed
        self._setup()

        # Start the JavaScript web application (dev or prod mode)
        self._start_web_application()

        # Set up 3D visualization
        self._initialize_3d_visualization()

        # Set up screen configuration
        self._initialize_screen()

    def _initialize_3d_visualization(self) -> None:
        """
        Initialize the 3D visualization components.
        """
        # Define the output file path relative to the script's directory
        output_path = self.base_path.parent / "public" / "models" / "ares" / "scene.glb"

        # Try to get the segment from the backend if available
        if hasattr(self.env.unwrapped, "backend") and hasattr(
            self.env.unwrapped.backend, "segment"
        ):
            self.segment = self.env.unwrapped.backend.segment
        else:
            self.segment = self.env.unwrapped.segment

        # Build and export the 3D scene
        self.builder = Segment3DBuilder(self.segment)
        self.builder.build_segment(
            output_filename=str(output_path),
            is_export_enabled=self.is_export_enabled,
        )

        # Note: For the purpose of beam animation, we consider BEAM_SOURCE_COMPONENT
        # as the origin of the particle beam source
        self.lattice_component_positions = OrderedDict({BEAM_SOURCE_COMPONENT: 0.0})
        self.lattice_component_positions.update(self.builder.component_positions)

        # Store position of lattice components to use in JS web-app beam animation
        self.component_positions = list(self.lattice_component_positions.values())

        # Data to be used to send data over WebSocket
        self.data.update(
            {"component_positions": self.component_positions, "segments": {}}
        )

    def _initialize_screen(self) -> None:
        """Initialize the screen configuration for beam visualization."""
        # Define screen
        self.screen_name = SCREEN_NAME

        # Try to get the segment from the backend if available
        if hasattr(self.env.unwrapped, "backend") and hasattr(
            self.env.unwrapped.backend, "segment"
        ):
            self.screen = getattr(self.env.unwrapped.backend.segment, self.screen_name)
        else:
            self.screen = getattr(self.env.unwrapped.segment, self.screen_name)

        self.screen_resolution = DEFAULT_SCREEN_RESOLUTION
        self.screen_pixel_size = DEFAULT_SCREEN_PIXEL_SIZE
        self.screen.binning = DEFAULT_SCREEN_BINNING
        self.screen.is_active = True

        # Obtain screen boundaries
        self.screen_boundary = self.get_screen_boundary()

        # Update visualization data with screen boundaries
        self.data.update(
            {
                "screen_boundary_x": float(self.screen_boundary[0]),
                "screen_boundary_y": float(self.screen_boundary[1]),
            }
        )

    @property
    def control_action(self):
        """
        Delegate control_action to WebSocketWrapper.

        Returns:
            The current control action from the WebSocketWrapper.
        """
        return self.env.control_action

    @control_action.setter
    def control_action(self, value):
        """
        Set control_action in WebSocketWrapper.

        Args:
            value: The control action value to set.
        """
        self.env.control_action = value

    @property
    def render_mode(self):
        """Get the render mode."""
        return self._render_mode

    @render_mode.setter
    def render_mode(self, value):
        """Set the render mode."""
        self._render_mode = value

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment, reset last_action, and run the simulation.

        Args:
            seed (Optional[int]): Seed for random number generation.
            options (Optional[Dict]): Additional reset options.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info.
        """
        # Reset action state
        self.last_action = np.zeros(5, dtype=np.float32)
        self.current_step = 0

        # Reset the underlying environment
        observation, info = self.env.reset(seed=seed, options=options)

        # Read screen image
        image = info["backend_info"]["screen_image"]
        image = image.T
        self.screen_reading = np.flip(image, axis=0)  # Flip along y-axis

        # Run simulation
        self._simulate()

        return observation, info

    def _initialize_particle_beam(self) -> None:
        """
        Initialize the incoming particle beam for simulation.

        Raises:
            ValueError: If the incoming particle beam cannot be initialized.
        """
        # Try to get the beam from the backend if available
        if hasattr(self.env.unwrapped, "backend") and hasattr(
            self.env.unwrapped.backend, "incoming"
        ):
            self.incoming_particle_beam = (
                self.env.unwrapped.backend.incoming.as_particle_beam(
                    num_particles=self.num_particles
                )
            )
        # Otherwise get it from the incoming_beam attribute
        elif hasattr(self.env.unwrapped, "incoming_beam"):
            self.incoming_particle_beam = (
                self.env.unwrapped.incoming_beam.as_particle_beam(
                    num_particles=self.num_particles
                )
            )
        else:
            raise ValueError(
                "Cannot initialize incoming particle beam. Neither backend.incoming "
                "nor incoming_beam attributes found in the unwrapped environment."
            )

        if self.incoming_particle_beam is None:
            raise ValueError(
                "Incoming particle beam is None. Check beam initialization."
            )

        # Log the initial beam state for debugging
        logger.info(
            f"Initialized incoming particle beam with {self.num_particles} particles."
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute a step in the environment and run the simulation.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: Observation, reward,
                terminated, truncated, and info.
        """
        # Update last_action with the action being applied
        self.last_action = action.copy()

        # Execute step in the underlying environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Read screen image
        image = info["backend_info"]["screen_image"]
        image = image.T
        self.screen_reading = np.flip(image, axis=0)  # Flip along y-axis

        # Run simulation
        self._simulate()

        info.update({"stop_simulation": self.data["stop_simulation"]})

        return observation, reward, terminated, truncated, info

    async def render(self):
        """
        Render the environment by preparing simulation data and broadcasting it
        via WebSocket.
        This method does not rely on the underlying environment's render method, as all
        visualization logic is handled by this wrapper.

        Note: The simulation data is already updated in step() or reset(),
        so we don't need to call _simulate() again here.
        """
        if self.render_mode != "human":
            return  # Skip rendering if not in human mode

        # Delegate to WebSocketWrapper for broadcasting the data
        await self.env.broadcast(self.data)

        # Add delay after broadcasting to allow animation to complete
        # before sending new data
        await asyncio.sleep(1.25)

    def close(self):
        """
        Close the wrapper and terminate the web application process.
        """
        # Terminate the web application process if it exists
        if self.web_process:
            try:
                self.web_process.terminate()
                self.web_process.wait(timeout=5)
                logger.info("Terminated JavaScript web application process.")
            except subprocess.TimeoutExpired:
                logger.warning("Forcibly killing web application process...")
                self.web_process.kill()

        # Close the underlying environment
        super().close()

    def get_screen_boundary(self) -> np.ndarray:
        """
        Computes the screen boundary based on resolution and pixel size.

        The boundary is calculated as half of the screen resolution multiplied
        by the pixel size, giving the physical dimensions of the screen
        in meters.

        Returns:
            np.ndarray: The screen boundary as a 2D numpy array [width, height]
            in meters.
        """
        return np.array(self.screen.resolution) / 2 * np.array(self.screen.pixel_size)

    def _setup(self):
        """
        Automates the setup process by running npm install to install dependencies.
        This should be run once to ensure the JavaScript dependencies are installed.
        """
        try:
            # Path to the node_modules directory
            node_modules_path = os.path.join(self.base_path.parent, "node_modules")

            # Check if package.json exists to confirm we are in the correct directory
            package_json_path = os.path.join(self.base_path.parent, "package.json")
            if not os.path.exists(package_json_path):
                raise FileNotFoundError(
                    f"{package_json_path} not found."
                    f" Make sure you are in the correct project directory."
                )

            # Check if node_modules exists and is not empty
            if os.path.exists(node_modules_path) and os.listdir(node_modules_path):
                logger.info("Dependencies are already installed. Skipping npm install.")
            else:
                logger.info("Running npm install...")
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=self.base_path.parent,  # Run in directory with package.json
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=(
                        True if sys.platform == "win32" else None
                    ),  # Only use shell=True on Windows
                )

                # Log the output for debugging purposes
                if result.returncode == 0:
                    logger.info("npm install completed successfully.")
                else:
                    logger.error(f"npm install failed with error: {result.stderr}")
                    raise RuntimeError(f"npm install failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Error during setup: {e}")
            raise

    def _start_web_application(self):
        """
        Start the JavaScript web application (Vite development server)
        in a background thread.
        """

        def run_web_server():
            try:
                # Determine the mode and load the appropriate .env file
                node_env = os.getenv("NODE_ENV", "production")
                if node_env == "production":
                    env_file = script_dir.parent / ".env.production"
                    # Load with override existing vars to ensure latest values
                    load_dotenv(
                        dotenv_path=env_file,
                        override=True,
                    )

                logger.debug(f"Running in mode: {node_env}")

                if node_env == "development":
                    # Development mode: Start Vite dev server
                    # Start Vite development server
                    cmd = [
                        "npx",
                        "vite",
                        "--host",
                        self.http_host,
                        "--port",
                        str(self.http_port),
                    ]
                    logger.debug(
                        f"Starting Vite dev server"
                        f" on http://{self.http_host}:{self.http_port}"
                    )
                else:
                    # Production mode: Start Express server (server.js)
                    dist_path = self.base_path.parent / "dist"
                    if not dist_path.exists():
                        raise FileNotFoundError(
                            f"Pre-built dist folder not found at {dist_path}"
                        )
                    cmd = ["node", "server.js"]
                    logger.debug(
                        f"Starting Express server (server.js)"
                        f" on http://{self.http_host}:{self.http_port}"
                    )

                self.web_process = subprocess.Popen(
                    cmd,
                    cwd=self.base_path.parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    # Pass environment variables (e.g., PORT from .env)
                    env=os.environ.copy(),
                    shell=(
                        True if sys.platform == "win32" else None
                    ),  # Only use shell=True on Windows
                )

                # Log output for debugging
                for line in self.web_process.stdout:
                    logger.debug(f"Vite stdout: {line.strip()}")
                for line in self.web_process.stderr:
                    logger.error(f"Vite stderr: {line.strip()}")

            except Exception as e:
                logger.error(f"Failed to start web application: {e}")
                # Consider raising the exception here for better error handling

        # Start the web server in a background thread
        self.web_thread = threading.Thread(target=run_web_server, daemon=True)
        self.web_thread.start()

        # Give the server a moment to start
        logger.debug(
            f"JavaScript web application setup initiated on "
            f"http://{self.http_host}:{self.http_port}"
        )

    def _simulate(self) -> None:
        """
        Calculate the positions of beam segments with dynamic angles.

        This method tracks the particle beam through each element in the segment,
        computing the positions of particles at each step. The beam travels along
        the x-axis, with position variations in the yz-plane. The simulation
        data is stored in self.data for later use in visualization.
        """
        # Reset segments data for this simulation step
        self.data["segments"] = {}
        segment_index = 0

        # Reinitialize the incoming particle beam to ensure a fresh start
        # (i.e. initial state) at AREASOLA1
        self._initialize_particle_beam()

        # Track beam through each lattice element
        references = [self.incoming_particle_beam]
        for element in self.segment.elements:
            # Track beam through this element
            # Use the output beam of the previous segment as the input
            # for the next lattice section
            outgoing_beam = element.track(references[-1])
            references.append(outgoing_beam)

            logger.info(f"Tracked beam through element {element.name}: {outgoing_beam}")

            # Only store particle positions for elements in lattice_component_positions
            if element.name in self.lattice_component_positions:
                # Extract particle positions
                x = -outgoing_beam.particles[:, 0]  # Column 0
                y = outgoing_beam.particles[:, 2]  # Column 2
                z = -outgoing_beam.particles[:, 4]  # Column 4

                # Note: In Cheetah, the coordinates of the particles are defined
                # by a 7-dimensional vector: x = (x, p_x, y, p_y, 𝜏, 1),
                # where 𝜏 = t - t_0 represents the time offset of a particle
                # relative to the reference particle.
                #
                # Since we use z to represent the `longitudinal position` of particles
                # in the beamline (instead of time offset), we flip the sign of 𝜏.
                #
                # This ensures that particles:
                # - `ahead` of the reference particle (bunch head) have `positive` z,
                # - `behind` the reference particle (bunch tail) have `negative` z.
                #
                # This sign convention aligns with spatial representations
                # of beam bunches, where a leading particle has a larger
                # longitudinal position z.
                #
                # Source:
                # https://cheetah-accelerator.readthedocs.io/en/latest/coordinate_system.html

                # Shift beam particles 3D position in reference to segment component
                positions = torch.stack(
                    [x, y, z + self.lattice_component_positions[element.name]], dim=1
                )

                # Compute the mean position of the bunch
                mean_position = positions.mean(dim=0, keepdim=True)

                # Scale the spread (deviation from mean) using spread_scale_factor
                spread_scaled = (
                    positions - mean_position
                ) * self.ws_env.spread_scale_factor

                # Scale the mean position using mean_scale_factor
                # Note: We only scale x and y components, leaving z unchanged
                mean_scaled = mean_position.clone()

                if element.name == SCREEN_NAME:
                    # Apply amplification only to x and y components (index 0 and 1),
                    # leaving z unchanged
                    mean_scaled[0, 0] = self._amplify_displacement(
                        x=mean_position[0, 0],
                        amplification_factor=self.ws_env.mean_scale_factor,
                    )  # x
                    mean_scaled[0, 1] = self._amplify_displacement(
                        x=mean_position[0, 1],
                        amplification_factor=self.ws_env.mean_scale_factor,
                    )  # y

                    # Combine scaled spread with scaled mean position
                    positions = spread_scaled + mean_scaled
                else:
                    positions = spread_scaled + mean_scaled

                # Store segment data
                self.data["segments"][f"segment_{segment_index}"] = {
                    "segment_name": element.name,
                    "positions": positions.tolist(),
                }

                # Update segment index
                segment_index += 1

        # Try to get the segment from the backend if available
        if hasattr(self.env.unwrapped, "backend"):  # TODO: Generalize for other ARES
            # Get screen pixel reading
            # info = self.env.unwrapped.backend.get_info()
            # img = self.env.unwrapped.backend.get_screen_image()
            # screen_reading = info["screen_image"]
            pass
        else:
            # screen_reading = self.segment.AREABSCR1.reading
            pass

        self.current_step += 1

        # Update meta info to include particle reading from segments
        self.data.update(
            {
                "screen_reading": self.screen_reading.tolist(),
                "bunch_count": self.current_step,
                "stop_simulation": self.env.stop_simulation,
            }
        )

    def _amplify_displacement(
        self, x: torch.Tensor, amplification_factor: float = 50.0
    ) -> torch.Tensor:
        """
        Apply linear amplification to a displacement value to enhance small changes.

        Args:
            x (torch.Tensor): The input displacement value (in meters).
            amplification_factor (float): The factor by which to amplify
                small displacements (e.g., 10 or 100).

        Returns:
            torch.Tensor: The amplified displacement value.
        """
        return torch.sign(x) * amplification_factor * torch.abs(x)
