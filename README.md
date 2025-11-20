  # MPPI Tutorial
  This repository provides lecture materials, example code, and tutorial videos for a **3-session introduction to Model Predictive Path Integral (MPPI) control**.
  The tutorial is designed for an graduate mobile robotics course.

  ---

  ## 📚 Tutorial Structure (3 Sessions)

  1. **Session 1 — Essential Concepts for Understanding MPPI** 👉 **[Session 1](https://youtu.be/u8wPyTtI8as)**
  2. **Session 2 — MPPI Algorithm Explanation and Derivtation** 👉 **[Session 2](https://youtu.be/w_NsuKs25Z8)**
  3. **Session 3 — Code Walkthrough and Practical Implementation** 👉 **[Session 3](https://youtu.be/-9ET4rlpoz8)**

    

  ---
  # ⚙️ Environment Setup

  Below is the recommended environment configuration for running the MPPI tutorial with **ROS 2 Humble on Ubuntu 22.04**.

### 1) Install JAX (CUDA 12)

JAX can be installed directly via pip:

```bash
pip install -U "jax[cuda12]"

### 2) Clone the Project into Your ROS2 Workspace
mkdir -p ~/mppi_ros2_ws/src
cd ~/mppi_ros2_ws/src
git clone <repository_url>

### 3) Build the Package
cd ~/mppi_ros2_ws
colcon build
source install/setup.bash
  ---

  # 🚀 Running the Example Code

  ```bash
ros2 launch rci_action_manager simulation.launch.py
  ```

  ---

