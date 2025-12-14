# Negative Space Imaging Viewer

A professional web-based DICOM viewer with integrated quantum processing and negative space analysis capabilities, built on the OHIF (Open Health Imaging Foundation) platform.

## Features

### 🖼️ Advanced Medical Imaging

- **DICOM Support**: Full DICOM file format support with metadata parsing
- **Multi-viewport**: Simultaneous viewing of multiple images/studies
- **Zoom & Pan**: Smooth image manipulation with mouse and keyboard controls
- **Windowing**: Adjustable window/level for optimal contrast
- **Measurements**: Built-in measurement tools (length, angle, ROI)

### 🔬 Quantum Processing Integration

- **Real-time Analysis**: Live quantum processing of loaded images
- **Edge Detection**: Quantum-enhanced edge detection algorithms
- **Pattern Recognition**: Advanced pattern analysis using quantum circuits
- **Performance Metrics**: Real-time processing performance monitoring

### 📏 Negative Space Analysis

- **Custom Measurements**: Specialized tools for negative space detection
- **Automated Analysis**: AI-powered analysis of measurement regions
- **Statistical Metrics**: Comprehensive statistical analysis of regions
- **Visualization**: Overlay visualization of analysis results

### 🎨 Professional UI/UX

- **Modern Interface**: Clean, medical-grade user interface
- **Responsive Design**: Works on desktop and tablet devices
- **Dark Theme**: Optimized for extended viewing sessions
- **Accessibility**: WCAG 2.1 AA compliant interface

## Architecture

```
web_viewer/
├── public/                 # Static assets
│   ├── index.html         # Main HTML template
│   └── favicon.ico        # Application icon
├── src/
│   ├── components/        # React components
│   │   ├── ViewerComponent.jsx          # Main OHIF viewer
│   │   ├── QuantumProcessingOverlay.jsx # Quantum results overlay
│   │   └── NegativeSpaceMeasurementTool.jsx # Custom measurement tool
│   ├── services/
│   │   └── ApiService.js   # Backend API integration
│   ├── App.jsx            # Main application component
│   ├── App.css            # Application styles
│   └── index.js           # Application entry point
├── package.json           # Dependencies and scripts
└── README.md             # This file
```

## Prerequisites

- **Node.js**: Version 18.0.0 or higher
- **npm**: Version 8.0.0 or higher (comes with Node.js)
- **Python Backend**: The quantum processing backend must be running
- **Modern Browser**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

## Installation

1. **Navigate to the web viewer directory:**

   ```bash
   cd web_viewer
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start the development server:**

   ```bash
   npm start
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

## Backend Integration

The viewer requires a running Python backend for quantum processing. Ensure the backend API is available at `http://localhost:8000/api`.

### Backend Endpoints

- `POST /api/quantum/process` - Quantum image processing
- `POST /api/segmentation/analyze` - Image segmentation
- `POST /api/negative-space/analyze` - Negative space analysis
- `GET /api/health` - Health check
- `GET /api/models` - Available models

## Usage

### Loading Images

1. Click the "Choose File" button in the control panel
2. Select a DICOM file (.dcm) or standard image file
3. The image will load in the main viewer area

### Quantum Processing

1. Load an image first
2. Click "Run Quantum Analysis" in the control panel
3. View results in the quantum overlay and results panel

### Segmentation Analysis

1. Load an image first
2. Click "Run Segmentation" in the control panel
3. Segmentation masks will be displayed as overlays

### Negative Space Measurements

1. Select the negative space measurement tool (if available)
2. Click and drag to create a measurement region
3. View analysis results in the measurements panel

## Configuration

### OHIF Configuration

The viewer is configured via `src/config/ohifConfig.js`:

```javascript
const config = {
  routerBasename: '/',
  showStudyList: false,
  extensions: [],
  modes: [],
  // ... additional configuration
};
```

### API Configuration

Backend API endpoints are configured in `src/services/ApiService.js`:

```javascript
const apiService = new ApiService('http://localhost:8000/api');
```

## Development

### Available Scripts

- `npm start` - Start development server
- `npm run build` - Create production build
- `npm test` - Run tests
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier

### Adding New Tools

1. Create a new tool component in `src/components/`
2. Register it with OHIF in the main App component
3. Add UI controls in the control panel

### Extending API Integration

1. Add new methods to `ApiService.js`
2. Update the App component to use new endpoints
3. Add corresponding UI elements

## Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build` directory.

## Deployment

### Docker Deployment

```bash
# Build the Docker image
docker build -t negative-space-viewer .

# Run the container
docker run -p 3000:80 negative-space-viewer
```

### Static Hosting

The built application can be served from any static web server:

```bash
# Using serve
npx serve -s build -l 3000

# Using nginx
# Copy build contents to nginx html directory
```

## Browser Support

- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

## Troubleshooting

### Common Issues

1. **Backend Connection Failed**

   - Ensure the Python backend is running on port 8000
   - Check CORS configuration in the backend
   - Verify network connectivity

2. **Images Not Loading**

   - Check file format (DICOM or standard image)
   - Verify file integrity
   - Check browser console for errors

3. **Quantum Processing Unavailable**
   - Ensure backend has quantum processing enabled
   - Check Google Cirq installation
   - Verify quantum hardware/software requirements

### Debug Mode

Enable debug logging by setting localStorage:

```javascript
localStorage.setItem('debug', 'ohif:*');
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **OHIF**: Open Health Imaging Foundation
- **Cornerstone.js**: Medical imaging library
- **Google Cirq**: Quantum computing framework
- **React**: UI framework

## Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section
