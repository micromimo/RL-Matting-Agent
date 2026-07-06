# RL-Matting-Agent
![Rust](https://img.shields.io/badge/Rust-2024__Edition-black?style=flat-square&logo=rust&logoColor=white)
![Tauri](https://img.shields.io/badge/Tauri-v2-24C8D8?style=flat-square&logo=tauri&logoColor=white)
![React](https://img.shields.io/badge/React-18%2B-61DAFB?style=flat-square&logo=react&logoColor=black)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.0%2B-38BDF8?style=flat-square&logo=tailwind-css&logoColor=white)
![License: MPL 2.0](https://img.shields.io/badge/License-MPL__2.0-brightgreen)

<img width="2048" height="1280" alt="UI-1演示" src="https://github.com/user-attachments/assets/81dd8e33-48ab-47c4-9260-f81c0ba536c4" />

https://github.com/user-attachments/assets/fedb6c5e-db2e-41e3-b4cb-9c794858ef6b

TauriApp(Rust+React).An image subject extraction tool that integrates traditional image processing methods, deep learning models (custom-trained), and semantic segmentation models (existing U2-Net), while supporting customizable processing workflows and parameters.

一個整合了傳統影像處理方法（Rust標準圖形處理庫）、深度學習模型（自己訓練的）、語意分割模型（現有U2模型），且支援自訂處理工序與參數的影像主體擷取工具。

# Deployment Guide

    Clone the repository:
    git clone https://github.com/micromimo/RL-Matting-Agent.git

    Prepare the assets:
        Place the ONNX format models into /src-tauri/target/debug/up
        Place the U2Net series models into /src-tauri/target/debug/models

    Run the development server:
      Navigate to the project root directory and run the following commands in your terminal:
        cd RL-Matting-Agent
        npm install
        npm run tauri dev

# 部署方法
* **终端中执行**：
git clone https://github.com/micromimo/RL-Matting-Agent.git

* **资源准备**:
将OONX格式的模型放入/src-tauri/target/debug/\_up_，U2系列模型放在/src-tauri/target/debug/models

* **终端中执行**：

  cd {项目根目录}

  npm install

  npm run tauri dev

