# Parallel DNA Sequence Matcher

A high-performance, parallelized implementation of DNA sequence matching algorithms (Smith-Waterman and KMP). This project demonstrates the acceleration of computational biology tasks using various parallel computing paradigms including OpenMP, MPI, and OpenCL, alongside a sequential baseline and an interactive graphical user interface.

## 🧬 Algorithms Implemented
*   **Smith-Waterman**: For local sequence alignment.
*   **Knuth-Morris-Pratt (KMP)**: For exact pattern matching.

## 🚀 Parallel Computing Frameworks
1.  **Sequential Baseline**: Standard CPU execution for performance comparison.
2.  **OpenMP**: Shared-memory multi-threading for multi-core CPUs.
3.  **MPI (Message Passing Interface)**: Distributed-memory parallelism for cluster computing.
4.  **OpenCL**: Heterogeneous parallel computing (GPUs and accelerators).
5.  **Interactive UI**: A Dear ImGui-based graphical interface for visualizing sequence matching and benchmarking different parallel hardware.

## 🛠️ Prerequisites

### Linux / WSL (Ubuntu/Debian-based) - **Recommended**
Install the required development tools and libraries:
```bash
sudo apt update
sudo apt install -y build-essential cmake \
    libomp-dev \
    openmpi-bin libopenmpi-dev \
    opencl-headers ocl-icd-opencl-dev pocl-opencl-icd \
    libglfw3-dev libgl1-mesa-dev libglew-dev xorg-dev
```

### Windows
*   **CMake** and a C++ compiler (like MSVC via Visual Studio).
*   **MS-MPI** (Download and install from the official Microsoft releases).
*   **OpenCL SDK** (Provided by your GPU driver vendor: NVIDIA, AMD, or Intel).

## ⚙️ Building the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dawoodshah04/Parallel-DNA-Sequence-Matcher.git
   cd Parallel-DNA-Sequence-Matcher
   ```

2. **Fetch the ImGui Dependency:**
   If you haven't already initialized the Dear ImGui dependency, clone it into the `external/imgui` folder:
   ```bash
   git clone -b docking https://github.com/ocornut/imgui.git external/imgui
   ```

3. **Configure and Build using CMake:**
   ```bash
   mkdir build
   cd build
   cmake ..
   make -j$(nproc)
   ```

## 🏃‍♂️ Running the Executables

Once compiled, the binaries will be available in your `build` directory.

*   **Sequential Baseline**: 
    ```bash
    ./dna_sequential
    ```
*   **OpenMP Version**: 
    ```bash
    ./dna_openmp
    ```
*   **MPI Version** (Replace `4` with your desired number of processes):
    ```bash
    mpirun -np 4 ./dna_mpi
    ```
*   **OpenCL Version**:
    ```bash
    ./dna_opencl
    ```
*   **Graphical UI**:
    *(Note: On WSL, requires WSLg or an X-Server like VcXsrv)*
    ```bash
    ./dna_ui
    ```

## 📄 License
See the [LICENSE](LICENSE) file for details.