# SatQuery AI 🛰️
### Agentic Vision-Language Assistant for Remote Sensing Analysis

**SatQuery AI** is an intelligent, query-driven agentic framework designed for high-level geographic reasoning. Unlike static AI models, SatQuery AI acts as a "Controller" that interprets natural language queries, selects specialized remote-sensing models, and combines their outputs to provide evidence-grounded, auditable responses. 

The system is specifically optimized for joint reasoning over **Single-Image**, **Bi-Temporal (Change)**, and **Cross-Modal (Optical + SAR)** data.

---

## 📜 Functional Scope & Mandatory Compliance
SatQuery AI is built to strictly adhere to the functional requirements defined by ISRO/SAC:

*   **Remote-Sensing Adaptation:** Utilizes visual components fine-tuned on multisensor datasets (BigEarthNet) and specialized benchmarks (RSVQA, VRSBench).
*   **Single-Image Intelligence:** Provides mandatory Visual Question Answering (VQA) combined with text-guided region grounding and scene description.
*   **Multi-Image Change Analysis:** Mandatory bi-temporal change description and change-based VQA using spatially corresponding image pairs.
*   **Cross-Modal Analysis:** Extracts and fuses complementary information from co-registered Optical/Multispectral and Synthetic Aperture Radar (SAR) imagery.
*   **Agentic Orchestration:** Automatically selects, sequences, and executes specialist tools based on input configuration and query intent.

---

## 🚀 Key Features

### 🧠 Agentic Orchestration
The system features a **Task-Routing Controller** that:
1.  Classifies the requested task from a natural language query.
2.  Validates input modality (Single vs. Paired) and format (GeoTIFF).
3.  Selects the optimal tool (e.g., ChangeFormer for change detection or Qwen2-VL for VQA).
4.  Generates an **Auditable Execution Trace** containing model names, parameters, and confidence scores.

### ⏱️ Bi-Temporal Change Engine
A specialized pipeline for detecting and describing geographic shifts over time:
*   **Geometric Alignment:** Sub-pixel image registration using Siamese feature descriptors.
*   **Radiometric Normalization:** Histogram matching to eliminate lighting noise between different acquisition dates.
*   **Change-VQA:** Explains *what* changed (e.g., "Built-up area increased by 12%") rather than just providing a heat map.

### 📡 Cross-Modal (Optical-SAR) Fusion
Overcomes the limitations of optical-only sensors:
*   **Structural Truth:** Uses SAR backscatter to identify water bodies and urban structures through cloud cover.
*   **All-Weather Reasoning:** Fuses specular reflectance data with multispectral context to answer queries like "Identify water bodies under this monsoonal cloud cover."

### 🔍 Evidence-Grounded Reasoning
Every response is backed by:
*   **Spatial Evidence:** Bounding boxes, change masks, and saliency maps.
*   **Scientific Metrics:** Signal-to-Noise Ratio (SNR), Registration Trust, and Spatial Coherence scores.

---

## 🛠 Technical Architecture

*   **Logic Framework:** Python-based Agentic Tool Selection.
*   **Cognitive Layer:** Qwen2-VL (4-bit NF4 Quantization) for VLM reasoning.
*   **Specialist Layer:** OpenCV, Rasterio, and Grounding DINO for signal processing and localization.
*   **Infrastructure:** Support for GeoTIFF/TIFF geospatial metadata.

---

## 📂 Repository Structure

```text
SatQuery_AI/
├── app.py                 # Streamlit-based Interactive GUI
├── src/                   # Core Logic Folder
│   ├── agent_engine.py     # Task Routing & Tool Orchestration
│   ├── BT_CM.py            # Bi-Temporal & Cross-Modal Specialist
│   ├── models_registry.py  # RS-VLM & Grounding Engine
│   └── geotiff_reader.py   # Geospatial Data Parser
├── data/                  # Input Directory (Place GeoTIFFs here)
├── outputs/               # Audit Logs and Generated Reports
└── tests/                 # Benchmark & Contract Testing Scripts

## 📖 How to Use

### 1. Installation
Ensure you have a Python 3.9+ environment with an NVIDIA GPU (min 6GB VRAM) for local inference.
```bash
git clone https://github.com/akashm-05/SatQuery_AI.git
cd SatQuery_AI
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your satellite images in the `/data` folder.
*   **Single Task:** Upload 1 GeoTIFF.
*   **Change Task:** Upload 2 GeoTIFFs (T1 and T2).
*   **SAR Task:** Upload 1 Optical and 1 SAR GeoTIFF.

### 3. Launch the Assistant
```bash
streamlit run app.py
```

### 4. Example Queries
Input these directly into the natural language box:
*   *"Describe the land-cover and major objects visible in this image."*
*   *"Locate all airplanes and highlight them with bounding boxes."*
*   *"What changed between these two dates, and where did the change occur?"*
*   *"Use the optical and SAR images together to identify built-up regions."*

---

## 📊 Auditable Execution Summary
To satisfy the ISRO/SAC evaluation criteria, the system generates a downloadable **Mission Audit Report**. 
Each report contains:
*   **Task Classification:** (e.g., `BITEMPORAL_CHANGE`)
*   **Specialist Tools Used:** (e.g., `ChangeFormer-V2`, `RS-VLM`)
*   **Confidence Information:** Mathematical trust scores based on signal coherence.
*   **Parameters:** Specific task parameters used during execution.
