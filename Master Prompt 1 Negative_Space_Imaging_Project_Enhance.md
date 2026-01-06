@workspace /new Act as a Distinguished System Architect and Lead Research Engineer. I require you to execute a comprehensive, three-tier technological upgrade for this repository, integrating state-of-the-art open-source advancements from late 2024/2025. Proceed with the following implementation plan autonomously, ensuring all code is production-grade, modular, and meticulously documented.

### Phase 1: Advanced AI Segmentation (UNet++)
Create a new Python module named `segmentation_unet_plus.py`.
- **Objective:** Implement the **UNet++ (Nested U-Net)** architecture for superior medical image segmentation.
- **Requirements:**
  - Define the model architecture using PyTorch or TensorFlow/Keras.
  - Include dense skip connections and deep supervision capabilities suitable for detecting fine details in "negative space" imaging (e.g., lesions in CT/MRI).
  - Provide a class structure with methods for `build_model()`, `train_step()`, and `predict_mask()`.

### Phase 2: Hybrid Quantum-Classical Kernel (Google Cirq)
Create a new Python module named `quantum_processor.py`.
- **Objective:** Establish a hybrid quantum-classical processing layer using **Google Cirq**.
- **Requirements:**
  - Implement a `QuantumImageEncoder` class that converts classical image data into quantum states.
  - Design a parameterized quantum circuit (PQC) capable of acting as an edge-detection or feature-extraction kernel.
  - Include a simulation function that runs this circuit on a quantum simulator and returns the processed feature map to the classical pipeline.

### Phase 3: Professional Visualization Interface (OHIF Integration)
Scaffold a new web-based viewer component in a folder named `web_viewer`.
- **Objective:** Integrate **OHIF Viewer** core components to visualize the processed DICOM data.
- **Requirements:**
  - Create a basic React component structure that utilizes OHIF's extension mechanism.
  - Write a configuration script that connects this viewer to the Python backend's output API.
  - Ensure `package.json` is updated with necessary React and OHIF dependencies.

**Execution Standard:**
- Update `requirements.txt` with `torch`/`tensorflow`, `cirq`, and `pydicom`.
- Use strict typing (`typing` module) and comprehensive docstrings for all Python functions.
- Ensure error handling is robust for medical-grade reliability.