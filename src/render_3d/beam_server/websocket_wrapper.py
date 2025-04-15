import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import websockets

# Set logging level based on environment
debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Setup logging with conditional log level
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Define constants at module level
DEFAULT_WS_HOST = "127.0.0.1"
DEFAULT_WS_PORT = 8081
DEFAULT_CONNECTION_TIMEOUT = 1.0
DEFAULT_SPREAD_SCALE_FACTOR = 15
DEFAULT_MEAN_SCALE_FACTOR = 10


class WebSocketWrapper(gym.Wrapper):
    """
    A Gym wrapper that enables WebSocket integration for communication
    with a Gym-based environment.
    Manages WebSocket server and client communication internally.
    """

    def __init__(
        self,
        env: gym.Env,
        ws_host: str = DEFAULT_WS_HOST,
        ws_port: int = DEFAULT_WS_PORT,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
        spread_scale_factor: float = DEFAULT_SPREAD_SCALE_FACTOR,
        mean_scale_factor: float = DEFAULT_MEAN_SCALE_FACTOR,
    ):
        """
        Initialize the WebSocketWrapper.

        Args:
            env (gym.Env): The underlying Gym environment.
            ws_host (str): WebSocket server hostname.
            ws_port (int): WebSocket server port.
            connection_timeout (float): Timeout for WebSocket connections in seconds.
            spread_scale_factor (float): A scaling factor applied to the spread
                of particles in the beam. This factor is used to adjust
                the particle distribution's spread across the simulation space.
            mean_scale_factor (float): A scaling factor applied to the mean position
                of the particles in the beam. This factor controls the central tendency
                or the average position of the particle distribution.
        """
        super().__init__(env)

        # Store host and port, with defaults if not defined in env.unwrapped
        self.ws_host = getattr(self.env.unwrapped, "host", ws_host)
        self.ws_port = getattr(self.env.unwrapped, "port", ws_port)
        self.connection_timeout = connection_timeout

        # Validate port number
        if not isinstance(self.ws_port, int) or not (1 <= self.ws_port <= 65535):
            logger.warning(
                f"Invalid port number {self.ws_port}. Defaulting to {self.ws_port}."
            )

        # WebSocket management attributes
        self.clients = set()
        self.connected = False
        self.server = None

        # Data to be transmitted to the JavaScript web application
        self.data = None
        self.spread_scale_factor = spread_scale_factor
        self.mean_scale_factor = mean_scale_factor

        self._control_action = np.zeros(5, dtype=np.float32)  # "no-op" action
        self.stop_simulation = False

        # Start the WebSocket server in a separate thread
        self._lock = threading.Lock()
        self._start_websocket_server()

    @property
    def control_action(self):
        """Get the current control action."""
        return self._control_action

    @control_action.setter
    def control_action(self, value):
        """Set the current control action."""
        self._control_action = value

    def _start_websocket_server(self):
        """Start the WebSocket server in a background thread."""

        def run_server():
            asyncio.run(self._run_server())

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        logger.debug(
            f"WebSocket server thread started on ws://{self.ws_host}:{self.ws_port}"
        )

    async def _run_server(self):
        """Run the WebSocket server."""
        self.server = await websockets.serve(
            self._handle_client,
            host=self.ws_host,
            port=self.ws_port,
        )
        logger.debug(f"WebSocket server running on ws://{self.ws_host}:{self.ws_port}")
        await self.server.wait_closed()

    async def _handle_client(
        self, websocket: websockets.WebSocketServerProtocol, path: str = None
    ):
        """Handle incoming WebSocket connections and messages."""
        with self._lock:
            self.connected = True
            self.clients.add(websocket)
        logger.debug("WebSocket connection established.")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.debug(f"Received data: {data}")

                    if "controls" in data:
                        # Update the control parameters based on the WebSocket data
                        controls = data.get("controls", {})

                        self.spread_scale_factor = controls.get("scaleBeamSpread", 0.0)
                        self.mean_scale_factor = controls.get("scaleBeamPosition", 0.0)
                        self.stop_simulation = controls.get("stopSimulation", False)

                        areamqzm1 = controls.get("AREAMQZM1", 0.0)
                        areamqzm2 = controls.get("AREAMQZM2", 0.0)
                        areamcvm1 = controls.get("AREAMCVM1", 0.0)
                        areamqzm3 = controls.get("AREAMQZM3", 0.0)
                        areamchm1 = controls.get("AREAMCHM1", 0.0)

                        # Store the control action as a numpy array
                        self.control_action = np.array(
                            [areamqzm1, areamqzm2, areamcvm1, areamqzm3, areamchm1],
                            dtype=np.float32,
                        )

                        logger.debug(
                            "Received controls: AREAMQZM1={}, AREAMQZM2={}, "
                            "AREAMCVM1={}, AREAMQZM3={}, AREAMCHM1={}".format(
                                areamqzm1,
                                areamqzm2,
                                areamcvm1,
                                areamqzm3,
                                areamchm1,
                            )
                        )
                except json.JSONDecodeError:
                    logger.error("Error: Received invalid JSON data.")
        except asyncio.exceptions.CancelledError:
            logger.debug("WebSocket task was cancelled.")
            raise
        except websockets.ConnectionClosed:
            logger.debug("WebSocket connection closed by client.")
        finally:
            with self._lock:
                self.clients.discard(websocket)
                if not self.clients:
                    self.connected = False
            logger.debug("Client cleanup completed.")

    async def broadcast(self, message: Dict):
        """Broadcast a message to all connected clients."""
        if not self.clients:
            return

        tasks = [client.send(json.dumps(message)) for client in self.clients]
        await asyncio.gather(*tasks, return_exceptions=True)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute a step in the environment using WebSocket control action if available.
        """
        if self.control_action is not None:
            action = self.control_action
            self.control_action = None  # Clear after use

        return self.env.step(action)

    def close(self):
        """Close the environment and WebSocket server."""
        if self.server:
            self.connected = False
            self.clients.clear()
            self.server.close()
            logger.debug("WebSocket server closed.")
        super().close()

    async def render(self):
        """Render the environment and broadcast data via WebSocket."""
        if hasattr(self.env, "render"):
            await self.env.render()  # Let the env prepare its state
            await self.broadcast(self.data)  # Broadcast the updated info

            # Add delay after broadcasting to allow animation to complete
            # before sending new
            await asyncio.sleep(1.25)
