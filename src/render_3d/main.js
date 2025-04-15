// Import the necessary THREE.js modules
import * as THREE from 'three';

// To allow for the camera to move around the scene
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// To allow for importing the .gltf file
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// For reflective surfaces
import { Reflector } from 'three/examples/jsm/objects/Reflector.js';

// For postprocessing effects
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

// To use materials from Three.js
import { Material } from 'three';

// Plotting
import Plotly from 'plotly.js-dist';

class SceneManager {
    constructor(containerId, language = 'en') {
        // General Properties (Language, UI Translations)
        // Retrieve language from URL or use default
        this.language = language;

        // Define text content for both languages
        this.translations = {
            de: {
                toggleModel: "Bildschirm umschalten"
            },
            en: {
                toggleModel: "Toggle Screen"
            }
        };

        // WebSocket properties
        this.ws = null;                       // WebSocket connection object, initially set to null
        this.reconnectAttempts = 0;           // Counter for tracking the number of reconnect attempts
        this.max_reconnect_attempts = 5;      // Maximum number of reconnect attempts before giving up
        this.reconnect_delay = 2000;          // Delay in milliseconds between reconnect attempts (2 seconds)
        this.isSceneReady = false;            // Flag to track whether the scene is fully loaded and ready
        this.newDataAvailable = false;        // Flag to track if new data has been received from the WebSocket

        // Initialize stop button state (False by default)
        this.stopState = false;

        // Particle System Properties
        this.particleCount = 1000;      // The total number of particles to be simulated, matching the Python code
        this.particles = [];            // Array to hold all particle instances for the simulation

        // Segment Properties
        this.segmentsCount = 7;         // Total number of lattice segments
        this.segmentDistances = [];     // Array holding the distance for each segment in the path
        this.segmentStartPoints = [];   // Array holding the starting positions of each segment
        this.totalPathLength = 0;       // Total length of the entire path, calculated from segment distances
        this.totalProgress = 0;         // Overall progress through all segments (from 0 to 1)

        // Animation Properties
        this.particleSpeed = 0.1;      // Units per frame (default: 0.1)
        this.currentData = null;       // Store latest WebSocket data
        this.animationRunning = true;  // Start with animation running

        // Scene Initialization
        this.scene = new THREE.Scene();
        this.scene.name = "Scene";

        // Setup core rendering components
        this.camera = this.setupCamera();
        this.renderer = this.setupRenderer(containerId);
        this.controls = this.setupOrbitalControls();
        this.composer = this.setupPostProcessing();

        // Model Management
        this.aresModel = null;
        this.undulatorModel = null;
        this.objectsToRender = ["ares", "undulator"];
        this.objToRender = this.objectsToRender[0]; // Default: "ares"

        // Raycasting and Interaction
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        // Scene Configuration
        this.setupLighting();
        this.createReflectivePlane();
        this.setupTargetPoint();
        this.createControlPanel();
        this.loadModels();

        // Create the 2D histogram window
        this.createPlotWindow();

        // Axes Helper (placed after scene setup for better readability)
        // X-axis → Red (+X direction), Y-axis → Green (+Y direction), and Z-axis → Blue (+Z direction)
        const axesHelper = new THREE.AxesHelper(0.25);
        axesHelper.name = "AxesHelper";
        this.scene.add(axesHelper);

        // Event Listeners
        this.setupEventListeners();
        this.updateButtonLabels();

        // Particle System Initialization
        this.createParticles();

        // WebSocket Setup (only after everything else is ready)
        this.setupWebSocket();

        // Start Animation Loop
        this.startAnimation();
    }

    // Scene Initialization
    setupCamera() {
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        // Set how far the camera will start from the 3D model
        camera.position.set(-1.5, 0.75, -1.5); // Initial camera position (x, y, z)

        camera.updateMatrixWorld();  // Apply rotation change

        return camera;
    }

    setupRenderer(containerId) {
        // Instantiate a new renderer and set its size
        const renderer = new THREE.WebGLRenderer({ alpha: false });  //Alpha: true allows for the transparent background
        renderer.setSize(window.innerWidth, window.innerHeight);

        // Add the renderer to the DOM
        document.getElementById(containerId).appendChild(renderer.domElement);
        return renderer;
    }

    setupOrbitalControls() {
        // Add orbit controls to the camera, enabling rotation and zoom functionality using the mouse
        const controls = new OrbitControls(this.camera, this.renderer.domElement);

        controls.target.set(0.0, 0.0, 2.0460399985313416); // Looking towards the center of the diagnostic screen
        controls.minDistance = 0;    // Minimum zoom distance (closer)
        controls.maxDistance = 40;   // Maximum zoom distance (farther)
        controls.minPolarAngle = 0;       // 0 radians (0 degrees) - Looking straight up (at the sky)
        controls.maxPolarAngle = Math.PI;   // π radians (180 degrees) - Looking straight down (at the ground)

        controls.update();  // Apply the change

        return controls;
    }

    setupLighting() {
        const topLight = new THREE.DirectionalLight(0xffffff, 1);  // (color, intensity)
        topLight.position.set(50, 50, 50); //top-left-ish
        topLight.castShadow = false;
        topLight.name = "TopDirectionalLight";
        this.scene.add(topLight);

        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        ambientLight.name = "AmbientLight";
        this.scene.add(ambientLight);
    }

    setupPostProcessing() {
        const composer = new EffectComposer(this.renderer);
        composer.addPass(new RenderPass(this.scene, this.camera));

        const params = {
            exposure: 1,
            strength: 0.25,
            radius: 1,
            threshold: 0.1
        };
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            params.strength,
            params.radius,
            params.threshold
        );
        composer.addPass(bloomPass);
        return composer;
    }

    setupTargetPoint() {
        // Create a small red rectangle (box) to mark the controls.target
        const targetGeometry = new THREE.BoxGeometry(0.004, 0.002, 0.003); // Width: 4 mm, Height: 2 mm, Depth: 3 mm
        const targetMaterial = new THREE.MeshBasicMaterial({
            color: 0xff0000,   // Red color
            transparent: true, // Enable transparency
            opacity: 0.5       // Set translucency (0 = fully transparent, 1 = fully opaque)
        });
        const targetPoint = new THREE.Mesh(targetGeometry, targetMaterial);

        // Set the position to match the controls target
        targetPoint.position.copy(this.controls.target);
        targetPoint.name = "TargetPoint";

        // Add it to the scene
        this.scene.add(targetPoint);
    }

    setupEventListeners() {
        // Window resize listener, allowing us to resize the window and the camera
        window.addEventListener("resize", () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Handle pagehide to allow back/forward cache (bfcache)
        window.addEventListener("pagehide", () => {
            if (window.myBroadcastChannel) {
                window.myBroadcastChannel.close();
                window.myBroadcastChannel = null;
            }

            if (this.websocket) {
                this.websocket.close();
            }
        });

        // Toggle model button
        const toggleModelButton = document.getElementById("toggle-model");
        if (toggleModelButton) {
            toggleModelButton.addEventListener("click", this.toggleModel.bind(this));
        }

        // Click event handling
        const delta = 4;
        let startX, startY;

        document.addEventListener("pointerdown", (event) => {
            startX = event.pageX;
            startY = event.pageY;
        });

        document.addEventListener("pointerup", (event) => {
            const diffX = Math.abs(event.pageX - startX);
            const diffY = Math.abs(event.pageY - startY);

            if (diffX < delta && diffY < delta) {
                // Logic to determine if the click was inside or outside the textbox
                this.handlePointerClick(event);
            }
        });

        // Close buttons
        const closeButtons = Array.from(document.getElementsByClassName("closeButton"));
        closeButtons.forEach((closeButton) =>
            closeButton.addEventListener("click", this.hideTextBoxes.bind(this))
        );
    }

    createReflectivePlane() {
        const geometry = new THREE.PlaneGeometry(200, 200);
        const groundMirror = new Reflector(geometry, {
            clipBias: 0.003,
            textureWidth: window.innerWidth * window.devicePixelRatio,
            textureHeight: window.innerHeight * window.devicePixelRatio,
            color: 0x3333333,
        });

        groundMirror.rotateX(-Math.PI / 2);
        groundMirror.position.y = -1.4;
        groundMirror.name = "Reflector";
        this.scene.add(groundMirror);
    }

    // Create control panel UI with sliders and reset button
    createControlPanel() {
        // Create the control panel container
        const panel = document.createElement('div');
        panel.style.position = 'absolute';
        panel.style.top = '20px';
        panel.style.left = '20px';
        panel.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        panel.style.padding = '6px';
        panel.style.borderRadius = '5px';
        panel.style.zIndex = '101';
        panel.style.color = '#fff';
        panel.style.fontFamily = 'Arial, sans-serif';
        panel.style.fontSize = '12px';
        panel.style.width = '200px';

        // Control panel title
        const title = document.createElement('h3');
        title.textContent = 'Control Panel';
        title.style.margin = '0 0 10px 0';
        title.style.fontSize = '14px';
        title.style.textAlign = 'center';
        panel.appendChild(title);

        // Define the controls and their properties
        // Note: Although five controls are defined here, your gym observation space is (4,). Adjust as needed.
        const controls = [
            { id: 'AREAMQZM1', type: 'k1', label: 'AREAMQZM1', min: -72, max: 72, step: 0.01, initial: 0 },
            { id: 'AREAMQZM2', type: 'k1', label: 'AREAMQZM2', min: -72, max: 72, step: 0.01, initial: 0 },
            { id: 'AREAMCVM1', type: 'angle', label: 'AREAMCVM1', min: -6.1782e-3, max: 6.1782e-3, step: 0.000123564, initial: 0.0 },
            { id: 'AREAMQZM3', type: 'k1', label: 'AREAMQZM3', min: -72, max: 72, step: 0.01, initial: 0 },
            { id: 'AREAMCHM1', type: 'angle', label: 'AREAMCHM1', min: -6.1782e-3, max: 6.1782e-3, step: 0.000123564, initial: 0.0 },
            { id: 'particleSpeed', type: 'speed', label: 'Particle Speed', min: 0.001, max: 1.0, step: 0.001, initial: 0.1 },
            { id: 'scaleBeamSpread', type: 'speed', label: 'Scale beam width', min: 1.0, max: 100.0, step: 1.0, initial: 15.0 },
            { id: 'scaleBeamPosition', type: 'speed', label: 'Scale beam position', min: 1.0, max: 101.0, step: 1, initial: 50.0 }
        ];

        // Create each slider element
        this.controlSliders = {};
        this.defaultValues = {}; // Store default values for reset
        controls.forEach(control => {
            const container = document.createElement('div');
            container.style.marginBottom = '8px';
            container.style.width = '100%'; // Ensure consistent width within the panel

            const label = document.createElement('label');
            label.textContent = control.label;
            label.htmlFor = control.id;
            label.style.display = 'block';
            label.style.marginBottom = '4px';

            const input = document.createElement('input');
            input.type = 'range';
            input.id = control.id;
            input.min = control.min;
            input.max = control.max;
            input.step = control.step;
            input.value = control.initial;
            input.style.width = '150px';

            // Display current value with fixed width
            const valueDisplay = document.createElement('span');
            valueDisplay.id = `${control.id}-value`;
            valueDisplay.textContent = control.initial;
            valueDisplay.style.marginLeft = '11px';
            valueDisplay.style.display = 'inline-block'; // Prevent width changes
            valueDisplay.style.minWidth = '10px';        // Ensure fixed width
            valueDisplay.style.textAlign = 'left';       // Align numbers neatly

            // Store default value
            this.defaultValues[control.id] = control.initial;

            input.addEventListener('input', () => {
                let displayValue = input.value;
                if (control.id === 'AREAMCVM1' || control.id === 'AREAMCHM1') {
                    displayValue = this.radToMrad(parseFloat(input.value)).toFixed(2);
                    valueDisplay.textContent = displayValue;
                } else {
                    valueDisplay.textContent = displayValue;
                }
                this.updateControls(control.id);
            });

            container.appendChild(label);
            container.appendChild(input);
            container.appendChild(valueDisplay);
            panel.appendChild(container);

            this.controlSliders[control.id] = input;
        });

        // Create reset button
        const resetButton = document.createElement('button');
        resetButton.textContent = 'Reset';
        resetButton.style.marginTop = '10px';
        resetButton.style.width = '40px'; // '50%'
        resetButton.style.height = '40px'; // Set the same height for a circle
        resetButton.style.padding = '0'; // No extra padding (prev '5px')
        resetButton.style.border = 'none';
        resetButton.style.borderRadius = '50%'; // Make it a circle (prev '3px')
        resetButton.style.display = 'flex'; // Ensure text is centered
        resetButton.style.alignItems = 'center';
        resetButton.style.justifyContent = 'center';
        resetButton.style.cursor = 'pointer';
        resetButton.style.backgroundColor = '#4885a8';
        resetButton.style.color = '#fff';
        resetButton.style.fontSize = '10px'; // '12px'

        // Reset function
        resetButton.addEventListener('click', () => {
            Object.keys(this.controlSliders).forEach(id => {
                this.controlSliders[id].value = this.defaultValues[id];
                document.getElementById(`${id}-value`).textContent = this.defaultValues[id];
            });
            // Explicitly update internal state after resetting sliders
            this.updateControls();
        });

        // Create Stop button
        const stopButton = document.createElement('button');
        stopButton.textContent = 'Stop';
        stopButton.style.width = '40px';
        stopButton.style.height = '40px';
        stopButton.style.borderRadius = '50%';
        stopButton.style.display = 'flex';
        stopButton.style.alignItems = 'center';
        stopButton.style.justifyContent = 'center';
        stopButton.style.fontSize = '12px';
        stopButton.style.backgroundColor = 'red';
        stopButton.style.color = '#fff';
        stopButton.style.border = 'none';
        stopButton.style.cursor = 'pointer';

        stopButton.addEventListener('click', () => {
            this.stopState = !this.stopState;  // Toggle between True/False
            stopButton.style.backgroundColor = this.stopState ? 'darkred' : 'red'; // Indicate state change

            // Send updated value over WebSocket
            this.updateControls();
        });

        // Common button styles
        const buttonStyle = {
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '12px',  // Ensure same font size
            padding: '0',
            margin: '0',        // Remove margin inconsistencies
            lineHeight: '1',    // Normalize text height inside buttons
            border: 'none',
            cursor: 'pointer',
        };

        // Apply styles to Reset button
        Object.assign(resetButton.style, buttonStyle);
        resetButton.style.backgroundColor = '#4885a8';
        resetButton.style.color = '#fff';

        // Apply styles to Stop button
        Object.assign(stopButton.style, buttonStyle);
        stopButton.style.backgroundColor = 'red';
        stopButton.style.color = '#fff';

        // Create button container
        const buttonContainer = document.createElement('div');
        buttonContainer.style.display = 'flex';
        buttonContainer.style.gap = '10px';
        buttonContainer.style.marginTop = '10px';
        buttonContainer.style.justifyContent = 'center'; // Aligns buttons to the left
        buttonContainer.style.width = '50%';

        // Append buttons to the button container
        buttonContainer.appendChild(resetButton);
        buttonContainer.appendChild(stopButton);

        // Append button container to the control panel
        panel.appendChild(buttonContainer);

        // Append the control panel to the container
        const containerEl = document.getElementById('container3D');
        if (containerEl) {
            containerEl.appendChild(panel);
        }
    }

    // Function to create and display the window in the top-right corner
    createPlotWindow() {
        // Create the container div for the floating window
        const plotWindow = document.createElement('div');
        plotWindow.style.position = 'fixed';
        plotWindow.style.top = '10px';
        plotWindow.style.right = '10px';
        plotWindow.style.width = '300px';   // Default: 400px
        plotWindow.style.height = '210px';  // Default: 300px
        plotWindow.style.backgroundColor = 'white';
        plotWindow.style.border = '2px solid black';
        plotWindow.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
        plotWindow.style.padding = '10px';
        plotWindow.style.border = 'none'; // Ensure no border
        plotWindow.style.backgroundColor = 'black'; // Match background color if needed
        plotWindow.style.zIndex = '1000';

        // Create the container for the plot
        this.plotContainer = document.createElement('div');
        this.plotContainer.id = 'dynamicPlot';
        this.plotContainer.style.width = '100%';
        this.plotContainer.style.height = '100%';
        this.plotContainer.style.minHeight = '250px'; // Ensure minimum height
        this.plotContainer.style.overflow = 'auto';

        // Add the container to the floating window
        plotWindow.appendChild(this.plotContainer);
        document.body.appendChild(plotWindow);

        // Enable dragging functionality
        let isDragging = false;
        let offsetX, offsetY;

        plotWindow.addEventListener('mousedown', (e) => {
            // Capture initial mouse position and the window's position
            isDragging = true;
            offsetX = e.clientX - plotWindow.getBoundingClientRect().left;
            offsetY = e.clientY - plotWindow.getBoundingClientRect().top;

            // Add a class or style to indicate dragging, if desired
            plotWindow.style.cursor = 'move';
        });

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                // Calculate new position of the window
                const xPos = e.clientX - offsetX;
                const yPos = e.clientY - offsetY;

                // Update the window's position
                plotWindow.style.left = `${xPos}px`;
                plotWindow.style.top = `${yPos}px`;
            }
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            plotWindow.style.cursor = 'default';
        });

        // Define initial layout with an empty plot with labels and grid
        const layout = {
            title: {
                text: "Camera Image", // Placeholder title
                font: { family: 'Arial, sans-serif', size: 16, color: 'white' },
                x: 0.5
            },
            xaxis: {
                title: {
                    text: '', // 'X position (mm)',
                    font: { size: 14, color: 'white' }
                },
                range: [-4, 4],  // Default range, will be updated dynamically
                showgrid: true,
                zeroline: false,
                visible: true,   // Optional: Hide axis
                tickfont: { color: 'white' },  // White tick labels
                gridcolor: 'white'  // Optional: Dim grid lines for better visibility
            },
            yaxis: {
                title: {
                    text: '', // 'Y position (mm)',
                    font: { size: 14, color: 'black' }
                },
                range: [-2, 2],  // Default range, will be updated dynamically
                showgrid: true,
                zeroline: false,
                visible: true,   // Optional: Hide axis
                tickfont: { color: 'white' },
                gridcolor: 'white'
            },
            margin: { l: 30, r: 0, t: 40 , b: 30 },
            autosize: true,
            paper_bgcolor: 'black',  // Background outside the plot area
            plot_bgcolor: 'black'    // Keep graph area transparen
        };

        // Initialize an empty heatmap trace to prevent layout conflicts
        const trace = {
            z: Array.from({ length: 510 }, () => Array(612).fill(null)),  // Empty 510x612 (y, x) placeholder grid with no color data
            x: [...Array(612).keys()],  // Column indices from 0 to 611 (x-axis)
            y: [...Array(510).keys()],  // Row indices from 0 to 509 (y-axis)
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: false,  // Do not include the color scale
        };

        // Initialize an empty plot with the defined layout
        Plotly.newPlot(this.plotContainer.id, [trace], layout);
    }

    updatePlot() {
        if (!this.currentData || !this.currentData.bunch_count ||
            !this.currentData.screen_boundary_x || !this.currentData.screen_boundary_y) {
            console.warn("No valid bunch count or boundary data available.");
            return;
        }

        // Extract 2D bunch count array
        const bunchCountData = this.currentData.screen_reading; // 510 x 612 array (y, x)
 
        // Define pixel centers based on screen boundaries
        const numRows = bunchCountData.length; // Number of rows (y) (510 pixels)
        const numCols = bunchCountData[0] ? bunchCountData[0].length : 0; // Number of columns (x) (612 pixels), with a check for empty array

        const xMin = -this.currentData.screen_boundary_x * 1e3; // Convert to mm
        const xMax = this.currentData.screen_boundary_x * 1e3;
        const yMin = -this.currentData.screen_boundary_y * 1e3;
        const yMax = this.currentData.screen_boundary_y * 1e3;

        // Generate axis values (centers of each pixel)
        const xValues = Array.from({ length: numCols }, (_, i) => xMin + (i + 0.5) * (xMax - xMin) / numCols);
        const yValues = Array.from({ length: numRows }, (_, i) => yMin + (i + 0.5) * (yMax - yMin) / numRows);

        // Update the heatmap histogram with new data
        const trace = {
            z: bunchCountData, // Already in (y, x) order
            x: xValues,
            y: yValues,
            type: 'heatmap',
            colorscale: 'Viridis',  // Use the 'Hot' colorscale for warm hues at high counts, alternatively,'Viridis'
            showscale: false,  // Remove the color scale
        };

        // Retrieve the current zoom state before updating
        const currentLayout = document.getElementById('dynamicPlot').layout;

        // If the user has zoomed in, preserve their zoom level
        const layout = {
            title: {
                text: `Camera Image`,
                font: {
                    family: 'Arial, sans-serif',
                    size: 16,
                    color: 'white'
                },
                //xref: 'paper',
                x: 0.5 // Centers the title
            },
            xaxis: {
                title: {
                    text: '', // 'X position (mm)',
                    font: {
                        family: 'Arial, sans-serif',
                        size: 14,
                        color: 'white'
                    }
                },
                range: currentLayout ? currentLayout.xaxis.range : [xMin, xMax],  // Preserve zoom
                showgrid: true,
                zeroline: false, // true
                visible: true,   // Optional: Hide axis
                tickfont: { color: 'white' },  // White tick labels
                gridcolor: 'white'  // Optional: Dim grid lines for better visibility
            },
            yaxis: {
                title: {
                    text: '', // 'Y position (mm)',
                    font: {
                        family: 'Arial, sans-serif',
                        size: 14,
                        color: 'white'
                    },
                    standoff: 20 // Moves label further from the axis
                },
                range: currentLayout ? currentLayout.yaxis.range : [yMin, yMax],  // Preserve zoom
                showgrid: true,
                zeroline: false, // true
                visible: true,   // Optional: Hide axis
                scaleanchor: 'x', // Ensures equal aspect ratio
                tickfont: { color: 'white' },  // White tick labels
                gridcolor: 'white'  // Optional: Dim grid lines for better visibility
            },
            margin: { l: 30, r: 0, t: 40 , b: 30 },
            autosize: true,
            paper_bgcolor: 'black',  // Background outside the plot area
            plot_bgcolor: 'black'  // Keep graph area transparent
        };

        // Render the heatmap with preserved zoom
        Plotly.react('dynamicPlot', [trace], layout);
    }

    // Initial creation of particles and adding to scene
    createParticles() {
        console.log('Create particles ...');

        const sphereGeometry = new THREE.SphereGeometry(0.0001, 8, 8); // Default: radius=0.001, widthSegments=8, heightSegments=8
        const material = new THREE.MeshBasicMaterial({
            color: 0x52FF4D, // Match original beam color
            transparent: true,
            opacity: 1.0,
            blending: THREE.AdditiveBlending
        });

        // Create particles
        for (let i = 0; i < this.particleCount; i++) {
            const sphere = new THREE.Mesh(sphereGeometry, material);

            this.particles.push({
                mesh: sphere,  // The THREE.Mesh object for rendering
                segment: i % this.segmentsCount, // Distribute particles across available segments
                index: i,
                progress: 0, // Math.random(), // Random initial progress
                currentPosition: new THREE.Vector3(),
                startPosition: new THREE.Vector3(),
                targetPosition: new THREE.Vector3()
            });

            sphere.name = "Sphere_" + i;

            this.scene.add(sphere);
        }
        console.log(`Total particles created: ${this.particles.length}`);
    }

    // Model Loading & Scene Management
    loadModels() {
        const loader = new GLTFLoader();

        // Ensure BASE_URL is correctly handled
        const baseUrl = import.meta.env.BASE_URL || '/'; // Default to '/' if undefined
        console.log('Base URL:', baseUrl); // Debug the value

        // Load Ares Model
        loader.load(
            `${baseUrl}models/ares/scene.glb`, // No leading '/' here; baseUrl provides it
            (gltf) => {
                this.aresModel = gltf.scene;
                this.scene.add(this.aresModel);

                this.scene.traverseVisible(object => {
                    if (object.material && this.getRelatedObjectNames(object).length > 0) {
                        object.material = object.material.clone();
                    }
                });

                // Initially hide Ares model if we're starting with Undulator
                if (this.objToRender === "undulator") {
                    this.aresModel.visible = false;
                }

                console.log('Ares model loaded');
            },
            undefined,
           (error) => console.error('Error loading Ares model:', error)
        );

        // Load Undulator Model
        loader.load(
            `${baseUrl}models/undulator/scene.glb`,
            (gltf) => {
                this.undulatorModel = gltf.scene;
                this.scene.add(this.undulatorModel);

                this.scene.traverseVisible(object => {
                    if (object.material && this.getRelatedObjectNames(object).length > 0) {
                        object.material = object.material.clone();
                    }
                });

                // Initially hide Undulator model if we're starting with Ares
                if (this.objToRender === "ares") {
                    this.undulatorModel.visible = false;
                }

                // Enable toggle button once both models are loaded
                const toggleModelButton = document.getElementById("toggle-model");
                if (toggleModelButton) toggleModelButton.disabled = false;

                console.log('Undulator model loaded');
            },
            undefined,
            (error) => console.error('Error loading Undulator model:', error)
        );
    }

    toggleModel() {
        this.hideTextBoxes();

        if (this.objToRender === "ares") {
            // Switch to Undulator
            this.objToRender = "undulator";
            this.aresModel.visible = false;
            this.undulatorModel.visible = true;
            // Toggle animationRunning to pause the loop at some point
            this.pauseAnimation();
        } else {
            // Switch to Ares
            this.objToRender = "ares";
            this.undulatorModel.visible = false;
            this.aresModel.visible = true;
            // Toggle animationRunning to start the loop at some point
            this.startAnimation();
        }
    }

    findObjectByName(name) {
        let foundObject = null;
        this.scene.traverse(object => {
            if (object.name === name) {
                foundObject = object;
            }
        });
        return foundObject;
    }

    radToMrad(rad) {
        return rad * 1000;
    }

    // Rendering & Animation
    startAnimation() {
        this.animationRunning = true;
        this.animate(); // restart the animation loop
    }

    pauseAnimation() {
        this.animationRunning = false; // pause the loop at some point
    }

    // Render the scene
    animate() {
        // Start the 3D rendering
        requestAnimationFrame(this.animate.bind(this));

        if (this.animationRunning && this.newDataAvailable) {
            const deltaTime = this.getElapsedTime();

            console.debug("deltaTime:", deltaTime);

            // Update total progress based on speed and time
            this.totalProgress += (deltaTime * this.particleSpeed) / this.totalPathLength;
            this.totalProgress = Math.min(this.totalProgress, 1.0); // Ensure we don't exceed 1.0 (100% progress)
            console.debug("totalProgress:", this.totalProgress);

            // Calculate actual distance traveled along the path
            const distanceTraveled = this.totalProgress * this.totalPathLength;
            console.debug("distanceTraveled:", distanceTraveled);

            // Find current segment and progress
            const { segmentIndex, segmentProgress } = this.findCurrentSegment(distanceTraveled);
            console.debug("segmentIndex:", segmentIndex, "segmentProgress:", segmentProgress);

            // Get position data for current and next segments
            let currentSegment = this.currentData.segments[`segment_${segmentIndex}`];
            let nextSegment = (segmentIndex === this.segmentsCount - 1)
                ? currentSegment
                : this.currentData.segments[`segment_${segmentIndex + 1}`];

            if (!currentSegment || (segmentIndex < this.segmentsCount - 1 && !nextSegment)) {
                console.error(`Segment data missing: current=${segmentIndex}, next=${segmentIndex + 1}`,
                               "Available segments:", Object.keys(this.currentData.segments));
                this.newDataAvailable = false;
                return;
            }

            console.debug("currentSegment:", currentSegment.segment_name, "nextSegment:", nextSegment.segment_name);

            // Update each particle
            this.particles.forEach((particle, i) => {
                const startPos = new THREE.Vector3(...currentSegment.positions[i]);
                const endPos = new THREE.Vector3(...nextSegment.positions[i]);

                // Interpolate position based on constant speed progress
                particle.mesh.position.lerpVectors(startPos, endPos, segmentProgress);

                console.debug(`Particle ${i} position:`, particle.mesh.position.z);

                // Keep particles fully visible across all segments
                particle.mesh.material.opacity = 1.0;
                particle.mesh.visible = true;

                // Only stop animation when we reach 100% of total progress
                if (this.totalProgress >= 1.0 && segmentIndex === this.segmentsCount - 1 && segmentProgress >= 1.0) {
                    this.newDataAvailable = false;
                    console.debug(`Animation completed for all ${this.segmentsCount} segments`);
                }
            });
        }

        this.renderer.render(this.scene, this.camera);
        this.composer.render();
    }

    calculateSegmentDistances() {
        // Get the z-axis positions array from component_positions
        const zPositions = this.currentData.component_positions;

        // Set first segment start point to 0 (beam starts at origin)
        this.segmentStartPoints = [];
        this.segmentDistances = [];
        this.totalPathLength = 0;

        // Add initial segment from 0 to the first component
        const initialDistance = zPositions[0]; // Distance from 0 to first position
        this.segmentDistances.push(initialDistance);
        this.totalPathLength += initialDistance;
        this.segmentStartPoints.push(this.totalPathLength);

        // Calculate distances between consecutive positions
        for (let i = 0; i < zPositions.length - 1; i++) {
            // Distance is simply the difference along z-axis between consecutive points
            const segmentDistance = zPositions[i + 1] - zPositions[i];

            // Store the segment distance
            this.segmentDistances.push(segmentDistance);

            // Update total path length
            this.totalPathLength += segmentDistance;

            // Next segment start point is the cumulative distance
            this.segmentStartPoints.push(this.totalPathLength);
        }

        console.debug("Segment start points:", this.segmentStartPoints);
        console.debug("Segment relative distances:", this.segmentDistances);
        console.debug("Total segment path length:", this.totalPathLength);
    }

    // Find which segment we're in based on distance traveled
    findCurrentSegment(distanceTraveled) {
        // Make sure segment distances and start points are calculated
        if (this.segmentDistances.length === 0 || this.segmentStartPoints.length === 0) {
            console.warn("Segment distances or start points not calculated!");
            return { segmentIndex: 0, segmentProgress: 0 };
        }

        console.debug("distanceTraveled:", distanceTraveled, "totalPathLength:", this.totalPathLength);

        // If we've exceeded the total path length, return the last segment at 100% progress
        if (distanceTraveled >= this.totalPathLength) {
            console.debug("Reached or exceeded total path length, returning last segment");
            return { segmentIndex: this.segmentsCount - 1, segmentProgress: 1.0 };
        }

        // Iterate through the segment start points to find the current segment
        for (let i = 0; i < this.segmentStartPoints.length - 1; i++) {

            const segmentStart = this.segmentStartPoints[i];
            const segmentEnd = this.segmentStartPoints[i + 1];
            console.debug("segment [start, end]", segmentStart, segmentEnd);

            // Check if the distance traveled is within the range of the current segment
            if (distanceTraveled >= segmentStart && distanceTraveled < segmentEnd) {
                // Calculate progress within this segment
                // Scaled from 0 to 1 within each segment
                const segmentDistance = this.segmentDistances[i+1];
                const distanceInSegment = distanceTraveled - segmentStart;
                const segmentProgress = distanceInSegment / segmentDistance;

                console.debug("distanceInSegment: ", distanceInSegment);
                console.debug("segmentDistance: ", segmentDistance)
                console.debug(`Segment ${i}: ${segmentStart} to ${segmentEnd}, progress: ${segmentProgress}`);

                // Return the index of the segment and the progress within it
                return {
                    segmentIndex: i,
                    segmentProgress: Math.min(Math.max(segmentProgress, 0), 1.0)
                };
            }
        }

        // If we've exceeded the total path length, return the last segment at 100% progress
        return { segmentIndex: this.segmentsCount - 1, segmentProgress: 0 };
    }

    getElapsedTime() {
        // Assuming you start the timing when the particle system is initialized or when the particle starts moving
        const now = performance.now(); // You could also use Date.now()
        return (now - this.startTime) / 1000;  // Returns time in seconds
    }

    // WebSocket setup
    setupWebSocket() {
        // Ensure connection status element exists before initializing connection
        this.ensureConnectionStatusElement();

        this.initializeScene()
            .then(() => this.connectWebSocket());
    }

    initializeScene() {
        return new Promise((resolve) => {
            const checkScene = () => {
                // Check if both models are loaded
                if (this.aresModel && this.undulatorModel) {
                    this.isSceneReady = true;
                    console.log('Scene is ready for WebSocket connection');
                    resolve();
                } else {
                    console.log('Waiting for scene initialization...');
                    setTimeout(checkScene, 100);
                }
            };
            checkScene();
        });
    }

    getWebSocketUrl() {
        // Setup default fallback
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

        const host = import.meta.env.VITE_PROXY_HOST_NAME || '127.0.0.1';
        const port = import.meta.env.VITE_BACKEND_PORT || '8081';

        // Use the environment variable if available, otherwise fallback to a default
        const wsUrl = import.meta.env.VITE_BACKEND_SERVER_URL || `${protocol}//${host}:${port}`;
        console.log('WebSocket URL from env:', wsUrl); // Debug the value

        return wsUrl;
    }

    connectWebSocket() {
        if (!this.isSceneReady) {
            console.log('Cannot connect WebSocket - scene not ready');
            return;
        }

        if (this.reconnectAttempts >= this.max_reconnect_attempts) {
            this.updateConnectionStatus(false, 'Connection failed after multiple attempts');
            return;
        }

        try {
            this.updateConnectionStatus(false, 'Connecting...');
            const url = this.getWebSocketUrl();
            console.log('Attempting to connect to:', url); // Log the URL being used

            this.ws = new WebSocket(url);

            this.ws.onopen = () => {
                console.log('WebSocket connected successfully');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true, 'Connected');
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data); // Assuming the data is in JSON format
                    console.log("WebSocket Data Update: Refreshing the scene and plot or restarting the animation!");
                    this.updateSceneFromWebSocket(data);

                    // Update the 2D plot dynamically
                    this.updatePlot();

                } catch (e) {
                    console.error('Error processing WebSocket message:', e);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false, `Connection error: ${error.message}`);
            };

            this.ws.onclose = (event) => {
                console.log('WebSocket closed:', event);
                this.updateConnectionStatus(false, 'Disconnected');
                this.reconnectAttempts++;

                // Only attempt reconnect if it wasn't an intentional close
                if (!event.wasClean) {
                    setTimeout(() => this.connectWebSocket(), this.reconnect_delay);
                }
            };
        } catch (e) {
            console.error('WebSocket connection error:', e);
            this.reconnectAttempts++;
            setTimeout(() => this.connectWebSocket(), this.reconnect_delay);
        }
    }

    updateSceneFromWebSocket(data) {
        console.debug(`Received WebSocket message: ${JSON.stringify(data, null, 2)}`);

        if (!data?.segments) {
            console.warn('Invalid WebSocket data received');
            return;
        }

        // Store current data
        this.currentData = data;

        console.debug("Available segments:", Object.keys(this.currentData.segments));

        // Only recalculate segment distances if the structure changes
        if (this.segmentDistances.length === 0) {  // this ensures we calculate it once
            this.calculateSegmentDistances();
        }

        // Reset progress state when new data arrives
        this.totalProgress = 0;

        // Initialize the start time per data update
        this.startTime = performance.now();

        // Flag that new data has arrived
        this.newDataAvailable = true;

        // Reset particle positions to segment_0
        const startSegment = this.currentData.segments['segment_0'];

        this.particles.forEach((particle, i) => {
            particle.mesh.position.set(...startSegment.positions[i]);
        });
        console.debug("Particles reset to AREASOLA1");
    }

    // Event Handling
    handlePointerClick(event) {
        const textBoxes = Array.from(document.getElementsByClassName("text-box"));
        const isClickInsideTextBox = textBoxes.some((textBox) =>
            textBox.contains(event.target)
        );

        // Hide the textbox if the click was outside
        if (!isClickInsideTextBox) {
            textBoxes.forEach((textBox) => {
                textBox.hidden = true;
            });
            this.setBloomByName([]);
        }

        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        this.raycaster.layers.set(0);

        const intersects = this.raycaster.intersectObjects(this.scene.children, true);

        if (intersects.length > 0) {
            const clickedObject = intersects[0].object;
            const allObjects = this.getRelatedObjectNames(clickedObject);
            // Set Bloom effects
            this.setBloomByName(allObjects);
            // Fills the text box with content corresponding to the clicked object's name,
            // adjusting for the correct language or context based on the object.
            this.fillTextBox(clickedObject.name);
        }
    }

    hideTextBoxes() {
        const textBoxes = Array.from(document.getElementsByClassName("text-box"));
        textBoxes.forEach((textBox) => {
            textBox.hidden = true;
        });
        this.setBloomByName([]);
    }

    fillTextBox(objectName) {
        const textBoxes = Array.from(document.getElementsByClassName("text-box"));
        textBoxes.forEach((textBox) => (textBox.hidden = true));

        // Mapping to handle different variations of object names
        const nameMapping = {
            'Electronsource': 'electronsource',
            'Circle': 'beam',
            'TorusCore': 'torus',
            'TorusShell': 'torus',
            'Quadruplecore': 'quadcore',
            'Quadruplecoils': 'quadcore',
            'Steerercoil': 'steering',
            'SteererCore': 'steering',
            'Screen': 'screen',
            'BodyWithLence': 'camera',
            'Cube': 'undulator'
        };

        // Find the matching base name
        const matchedKey = Object.keys(nameMapping).find(key =>
            objectName.startsWith(key)
        );

        if (matchedKey) {
            const boxId = `text-box-${nameMapping[matchedKey]}-${this.language}`;
            const textBox = document.getElementById(boxId);

            if (textBox) {
                textBox.hidden = false;
            }
        }
    }

    updateButtonLabels() {
        const toggleModelButton = document.getElementById("toggle-model");
        if (toggleModelButton) {
            toggleModelButton.textContent = this.translations[this.language].toggleModel;
        }
    }

    // Gather slider values, map them, and send over WebSocket
    updateControls(changedControlId = null) {
        if (!this.controlSliders) return;

        // Gather all current slider values into a dictionary.
        const controlValues = {
            AREAMQZM1: parseFloat(this.controlSliders['AREAMQZM1'].value),
            AREAMQZM2: parseFloat(this.controlSliders['AREAMQZM2'].value),
            AREAMCVM1: parseFloat(this.controlSliders['AREAMCVM1'].value),
            AREAMQZM3: parseFloat(this.controlSliders['AREAMQZM3'].value),
            AREAMCHM1: parseFloat(this.controlSliders['AREAMCHM1'].value),
            particleSpeed: parseFloat(this.controlSliders['particleSpeed'].value),
            scaleBeamSpread: parseFloat(this.controlSliders['scaleBeamSpread'].value),
            scaleBeamPosition: parseFloat(this.controlSliders['scaleBeamPosition'].value),
            stopSimulation: this.stopState  // Include stop button state
        };

        // Always update particleSpeed to match the slider value
        this.particleSpeed = controlValues.particleSpeed;

        // If a specific control changed, log it
        if (changedControlId) {
            const slider = this.controlSliders[changedControlId];

            if (changedControlId === 'stopSimulation') {
                controlValues[changedControlId] = newValue; // Directly assign the stop state
            } else if (changedControlId === 'particleSpeed') {
                // Update particleSpeed directly when its slider changes
                this.particleSpeed = parseFloat(slider.value);
            } else if (changedControlId === 'AREAMCVM1' || changedControlId === 'AREAMCHM1') {
                // Determine the mapping function based on the control id.
                // Assuming controls with 'angle' in their label (or specific ids).
                controlValues[changedControlId] = parseFloat(slider.value);
            } else {
                controlValues[changedControlId] = parseFloat(slider.value);
            }
        } else {
            // Update all controls if no specific id is provided.
            const controlValues = {
                AREAMQZM1: parseFloat(this.controlSliders['AREAMQZM1'].value),
                AREAMQZM2: parseFloat(this.controlSliders['AREAMQZM2'].value),
                AREAMCVM1: parseFloat(this.controlSliders['AREAMCVM1'].value),
                AREAMQZM3: parseFloat(this.controlSliders['AREAMQZM3'].value),
                AREAMCHM1: parseFloat(this.controlSliders['AREAMCHM1'].value),
                particleSpeed: parseFloat(this.controlSliders['particleSpeed'].value),
                scaleBeamSpread: parseFloat(this.controlSliders['scaleBeamSpread'].value),
                scaleBeamPosition: parseFloat(this.controlSliders['scaleBeamPosition'].value)
            };
        }

        // If WebSocket is open, send the control values (excluding particleSpeed as it's local)
        const wsData = { controls: { ...controlValues } };
        delete wsData.controls.particleSpeed; // Remove particleSpeed from WebSocket data

        // If WebSocket is open, send the control values
        console.debug(`Sending update for ${changedControlId ? changedControlId : 'all controls'}:`, JSON.stringify(wsData));

        // Confirm the WebSocket is connected before sending updates:
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(wsData));
        } else {
            console.warn("WebSocket not ready. Message not sent.");
        }
    }

    // Utility Methods
    getRelatedObjectNames(clickedObject) {
        const objects = [
            [ // Gun
                "Electronsource"
            ],
            [ // Quadrupole
                "Quadruplecore", "Quadruplecoils",
                "Quadruplecore_1", "Quadruplecore_2", "Quadruplecore_3",
                "Quadruplecoils_1", "Quadruplecoils_2", "Quadruplecoils_3",
            ],
            [ // Beam
                "Circle", "Circle002"
            ],
            [ // Dipole
                "VerticalCorrectorcoil", "VerticalCorrectorcore",
                "Steerercoil_1", "Steerercoil_2", "SteererCore_1", "SteererCore_2"
            ],
            [ // Diagnostic screen
                "Screen"
            ],
            [ // Cavities
                "TorusCore", "TorusShell", "Cavities"
            ],
            [ // Camera
                "BodyWithLence", "BodyWithLence_1", "BodyWithLence_2",
                "BodyWithLence_3", "BodyWithLence_4",
            ],
            [ // Undulator
                "Cube002", "Cube003", "Cube018", "Cube015", "Cube016", "Cube017",
                "Cube014", "Cube013", "Cube012", "Cube001", "Cube004", "Cube005",
                "Cube008", "Cube007", "Cube006", "Cube009", "Cube010", "Cube011"
            ]
        ];

        return objects.find((object) => object.includes(clickedObject.name)) || [];
    }

    setBloomByName(objectNames) {
        this.scene.traverseVisible((object) => {
            if (!object.material) return;

            if (objectNames.includes(object.name)) {
                object.material.emissive.setHex(0xffffff);
                if (object.name !== "Circle" && object.name !== "Circle002") {
                    object.material.emissiveIntensity = 0.3;
                }
            } else {
                if (object.name === "Circle" || object.name === "Circle002") {
                    object.material.emissive.setHex(0x52FF4D);
                } else {
                    object.material.emissiveIntensity = 0;
                }
            }
        });
    }

    // Create connection status element if it doesn't exist
    ensureConnectionStatusElement() {
        let statusElement = document.getElementById('websocket-status');
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.id = 'websocket-status';
            statusElement.style.position = 'absolute';
            //statusElement.style.top = '10px';

            statusElement.style.bottom = '10px';
            statusElement.style.left = '50%';
            statusElement.style.transform = 'translateX(-50%)';
            statusElement.style.padding = '5px 10px';
            statusElement.style.borderRadius = '4px';
            statusElement.style.fontFamily = 'monospace';
            statusElement.style.zIndex = '100';
            statusElement.style.maxWidth = '80%';
            statusElement.style.wordWrap = 'break-word';

            // Add to the container
            const container = document.getElementById('container3D');
            if (container) {
                container.appendChild(statusElement);
            }
        }
    }

    updateConnectionStatus(status, message) {
        const statusElement = document.getElementById('websocket-status');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = status ? 'connected' : 'disconnected';
            statusElement.style.color = status ? '#006400' : '#640000';  // 'green' and 'red', respectively
            statusElement.style.backgroundColor = status ? 'rgba(0, 255, 0, 0.2)' : 'rgba(255, 0, 0, 0.2)';
        } else {
            console.warn('WebSocket status element not found');
        }
    }
}

// Initialize the scene when the page loads
window.addEventListener('DOMContentLoaded', () => {
    // Retrieve the URL parameters from the current window's location
    const urlParams = new URLSearchParams(window.location.search);

    // Get the 'language' parameter from the URL, defaulting to 'de' (German) if not present
    const language = urlParams.get('language') || 'de';
    new SceneManager('container3D', language);
});

export default SceneManager;
