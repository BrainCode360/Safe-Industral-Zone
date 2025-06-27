# 🚧 Industrial Forklift Safety AI System
<p align="center">

🔒 Ensuring Person Safety, Collision Avoidance & Workplace Security in Industrial Warehouses
![git_pic](https://user-images.githubusercontent.com/19869426/193572085-84af4390-dfe1-4c7c-a16e-fd7e2d237bee.png)
<!-- README.md for Safe Industrial Zone AI System -->

<p align="center">
  <img src="docs/assets/forklift.png" alt="Forklift Safety AI System Banner" width="10%">
</p>

<h2 align="center">
  🚧 AI-Powered Forklift Safety System for Industrial Warehouses
</h2>

<p align="center">
  Real-time person detection, red-zone collision alerts, and portal-based safety reporting to secure your workplace.
</p>

<p align="center">
  <a href="https://github.com/BrainCode360/Safe-Industral-Zone/actions">
    <img src="https://img.shields.io/github/last-commit/BrainCode360/Safe-Industral-Zone" alt="Build Status">
  </a>
  <a href="https://img.shields.io/github/v/release/BrainCode360/Safe-Industral-Zone">
    <img src="https://img.shields.io/github/v/release/BrainCode360/Safe-Industral-Zone" alt="Latest Release">
  </a>
  <a href="https://img.shields.io/github/last-commit/BrainCode360/Safe-Industral-Zone">
    <img src="https://img.shields.io/github/last-commit/BrainCode360/Safe-Industral-Zone" alt="Last Commit">
  </a>
  <a href="https://img.shields.io/github/license/BrainCode360/Safe-Industral-Zone">
    <img src="https://img.shields.io/github/license/BrainCode360/Safe-Industral-Zone" alt="License">
  </a>
</p>

---

## 📌 Overview

**Safe Industrial Zone** is an embedded AI solution deployed on forklifts in busy industrial environments to **ensure human safety, collision avoidance, and incident tracking**. Built with **Jetson Xavier NX**, **TensorRT**, and **OpenCV**, the system provides:

- ✅ 360° perception with 4 wide-angle cameras (120° FOV each)
- ✅ Red-zone monitoring with a defined **5-meter safety perimeter**
- ✅ Instant alerts to drivers using HMI interfaces (buzzer/LED)
- ✅ Logs all safety events to a centralized **accident avoidance portal**

This system is **live in production**, helping prevent accidents and ensure compliance with warehouse safety policies.

---

## ✨ Features

| Feature | Status |
|--------|--------|
| 🧍 Person detection and vehicle detection | ✅ Implemented |
| 🔴 5-meter Red Zone dynamic perimeter | ✅ Implemented |
| 🎥 Multi-camera (4x) 120° coverage | ✅ Implemented |
| ⚠️ Real-time driver alerts (LED + buzzer) | ✅ Implemented |
| 📊 Safety event logging with image snapshot | ✅ Implemented |
| ☁️ Cloud-based safety portal dashboard | ✅ Implemented |
| 🚀 Runs on Jetson Xavier NX (TensorRT + CUDA) | ✅ Optimized & tested |
| 🔌 Easy power integration, industrial deploy-ready | ✅ Field-tested |

---

## 🧠 Technology Stack

| Stack | Details |
|-------|---------|
| 🔋 Hardware | Jetson Xavier NX, 4x wide-angle USB cameras, 12V DC input |
| ⚙️ Software | Python 3.8+, OpenCV 4.2, TensorRT 7.1, CUDA 10.2 |
| 📶 Network | MQTT/REST APIs to send alerts to portal |
| 📷 AI Model | Optimized YOLOv8 model for person & object detection |
| 📁 Portal | Web dashboard (React + Firebase) for analytics |

---

## 🗂️ System Design Overview

### 🌐 Safety Portal Integration
Every red-zone violation or safety warning is logged remotely for review by safety officers.

### 📝 Log details include:

- 📅 Timestamp
- 🚜 Forklift ID
- 🧍 Object Detected (e.g. person, vehicle)
- 📸 Snapshot Image
- 📍 GPS/Zone (optional)

✅ These logs power heatmaps, trend analysis, and near-miss reports via the central portal.

### 🎬 Demo Preview
<p align="center"> <img src="https://raw.githubusercontent.com/BrainCode360/Safe-Industral-Zone/main/docs/assets/Zone_Violation.gif" alt="Forklift Safety AI Demo" width="80%"> </p>


