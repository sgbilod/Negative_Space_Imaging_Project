# Negative Space Mobile Application Framework

## Overview

The Negative Space Mobile Application Framework is a React Native implementation for field verification of negative space signatures. This framework allows users to capture, verify, and interact with negative space signatures using a mobile device. It communicates with the main Python backend through a custom API bridge.

## Architecture

The mobile application framework follows a layered architecture:

1. **Presentation Layer**
   - React Native components and screens
   - UI state management
   - Navigation and routing

2. **Business Logic Layer**
   - API integration
   - Signature verification
   - Blockchain integration
   - Local storage management

3. **Data Access Layer**
   - Mobile-Python bridge
   - WebSocket and REST communication
   - Native modules for performance-critical operations

4. **Device Integration Layer**
   - Camera access
   - Augmented reality components
   - Secure storage

## Key Components

### 1. API Bridge (NegativeSpaceAPI.js)

The API bridge provides a communication layer between the mobile app and the Python backend. It supports:

- REST API for basic operations
- WebSocket for real-time data exchange
- Fallback mechanisms for offline operation
- Native module bindings for performance-critical operations

### 2. Mobile Bridge Server (mobile_bridge.py)

A Python server that acts as an intermediary between the mobile app and the core Negative Space backend:

- Exposes REST and WebSocket endpoints
- Handles image processing for signature extraction
- Manages blockchain interactions
- Provides visualization data for AR display

### 3. React Native Screens

- **HomeScreen**: Main dashboard for the application
- **CameraScreen**: Interface for capturing negative space using the device camera
- **VerificationScreen**: Displays signature verification results
- **BlockchainScreen**: Shows blockchain registration details
- **SettingsScreen**: Configuration options for the application

### 4. Native Modules (Future Implementation)

Planned native modules for optimized performance:

- **NegativeSpaceExtractor**: C++/Swift implementation of signature extraction algorithms
- **OfflineVerifier**: Local verification capabilities without server connection
- **ARVisualizer**: Native AR visualization of negative spaces

## Communication Flow

1. User captures a negative space image using the device camera
2. Image is sent to the server for processing (or processed locally if native module is available)
3. Extracted signature is verified against known signatures
4. Verification results are displayed to the user
5. Optional blockchain verification provides additional trust

## Offline Capabilities

The mobile framework supports offline operation:

- Local signature extraction using native modules
- Cached signature database for offline verification
- Queued blockchain operations for later execution
- Sync mechanisms for reconnection

## Security Considerations

- All communication with the server is encrypted
- Sensitive data is stored in secure storage
- Multi-signature verification enhances security
- Blockchain provides immutable verification records

## Future Enhancements

1. **Enhanced AR Visualization**
   - Real-time negative space overlay
   - Interactive 3D visualization of signature features

2. **Advanced Camera Controls**
   - Guided capture assistance
   - Automatic lighting adjustment
   - Multi-frame signature extraction

3. **Field Verification Tools**
   - Geographic location tagging
   - Timestamp verification
   - Environmental condition recording

4. **Decentralized Verification**
   - Peer-to-peer signature verification
   - Distributed signature database
   - Consensus-based verification

## Getting Started

### Prerequisites

- Node.js 14+
- React Native CLI
- Python 3.8+ (for the mobile bridge server)
- Android Studio or Xcode for platform-specific development

### Installation

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Run on a device or emulator:
   ```bash
   npm run android
   # or
   npm run ios
   ```

4. Start the mobile bridge server:
   ```bash
   python mobile/api/mobile_bridge.py
   ```

## Integration with Main Project

The mobile application framework integrates with the main Negative Space Imaging Project through:

1. Shared signature extraction algorithms
2. Compatible verification protocols
3. Unified blockchain integration
4. Consistent data formats and APIs

This ensures that signatures created or verified on the mobile device are fully compatible with the main system.
