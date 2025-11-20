  # MPPI Tutorial
  This repository provides lecture materials, example code, and tutorial videos for a **3-session introduction to Model Predictive Path Integral (MPPI) control**.
  The tutorial is designed for an undergraduate/graduate mobile robotics course.

  ---

  ## 📚 Tutorial Structure (3 Sessions)

  1. **Session 1 — Introduction & Fundamental Concepts**
  2. **Session 2 — MPPI Algorithm and Derivation**
  3. **Session 3 — Code Walkthrough & Review of Recent Research**

  ---

  # ⚙️ Environment Setup

  Below is the recommended environment configuration for running the MPPI tutorial examples.

  ### 1) Create & Activate Conda Environment

  ```bash
  conda create -n mppi_tutorial python=3.10
  conda activate mppi_tutorial
  ```

  ### 2) Install Dependencies

  ```bash
  conda install matplotlib
  conda install pytorch=2.5.1 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
  conda install -c rapidsai -c nvidia -c conda-forge -c defaults \
      cuml python=3.10 cudatoolkit=11.8
  ```

  ---

  # 🚀 Running the Example Code

  ```bash
  git clone <repository_url>
  cd <repository_name>

  python3 examples/quadrotor3d_run.py
  ```

  ---

  # 🎥 Tutorial Videos

  You can watch the full tutorial series here:

  📹 **MPPI Tutorial Lecture Video**  
  👉 (Insert your YouTube link here)

  ---
