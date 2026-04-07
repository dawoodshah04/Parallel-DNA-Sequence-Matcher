cmake_minimum_required(VERSION 3.20)
project(ParallelDNASequenceMatcher CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Common static library ──────────────────────────────────────────────────
add_library(dna_common STATIC
    common/dna_utils.cpp
    common/smith_waterman.cpp
    common/kmp.cpp
)
target_include_directories(dna_common PUBLIC include)

# ── Sequential baseline ────────────────────────────────────────────────────
add_executable(dna_sequential sequential/main_sequential.cpp)
target_link_libraries(dna_sequential PRIVATE dna_common)

# ── OpenMP ─────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)
add_executable(dna_openmp openmp/main_openmp.cpp)
target_link_libraries(dna_openmp PRIVATE dna_common OpenMP::OpenMP_CXX)

# ── MPI ────────────────────────────────────────────────────────────────────
if(WIN32)
    find_path(MSMPI_INCLUDE_DIR mpi.h
        HINTS "$ENV{MSMPI_INC}"
              "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
    find_library(MSMPI_LIB msmpi
        HINTS "$ENV{MSMPI_LIB64}"
              "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")
    if(NOT MSMPI_INCLUDE_DIR OR NOT MSMPI_LIB)
        message(WARNING "MS-MPI not found — dna_mpi will be skipped. "
                        "Install from https://github.com/microsoft/Microsoft-MPI/releases")
    else()
        add_executable(dna_mpi mpi/main_mpi.cpp)
        target_include_directories(dna_mpi PRIVATE ${MSMPI_INCLUDE_DIR})
        target_link_libraries(dna_mpi PRIVATE dna_common ${MSMPI_LIB})
    endif()
else()
    # WSL2 / Linux — OpenMPI
    find_package(MPI REQUIRED)
    add_executable(dna_mpi mpi/main_mpi.cpp)
    target_link_libraries(dna_mpi PRIVATE dna_common MPI::MPI_CXX)
endif()

# ── OpenCL ─────────────────────────────────────────────────────────────────
find_package(OpenCL REQUIRED)
add_executable(dna_opencl opencl/main_opencl.cpp)
target_link_libraries(dna_opencl PRIVATE dna_common OpenCL::OpenCL)

# Copy .cl kernel files next to the executable after build
add_custom_command(TARGET dna_opencl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/opencl/sw_kernel.cl
        ${CMAKE_SOURCE_DIR}/opencl/kmp_kernel.cl
        $<TARGET_FILE_DIR:dna_opencl>
    COMMENT "Copying OpenCL kernel files to build directory"
)

# ── CUDA ───────────────────────────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    add_executable(dna_cuda 
        cuda/main_cuda.cu
        cuda/kernels.cu
    )
    target_link_libraries(dna_cuda PRIVATE dna_common)
    target_include_directories(dna_cuda PRIVATE include)
    
    # Set CUDA architecture (adjust based on your GPU)
    # Auto-detect if possible, otherwise use common architectures
    set_target_properties(dna_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "50;60;70;75;80;86;89"  # Covers most GPUs from Maxwell to Ada
    )
    
    message(STATUS "CUDA support enabled - dna_cuda target added")
else()
    message(WARNING "CUDA compiler not found — dna_cuda will be skipped. "
                    "Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
endif()

# ── UI — Dear ImGui + GLFW + OpenGL3 ──────────────────────────────────────
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)

if(NOT EXISTS ${IMGUI_DIR}/imgui.h)
    message(WARNING
        "ImGui submodule not found at ${IMGUI_DIR}.\n"
        "Run: git submodule update --init --recursive\n"
        "dna_ui target will be skipped.")
else()
    set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )

    add_executable(dna_ui
        ui/main_ui.cpp
        ui/sw_visualizer.cpp
        ui/benchmark_panel.cpp
        ui/pattern_search.cpp
        ${IMGUI_SOURCES}
    )
    target_include_directories(dna_ui PRIVATE
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        include
        ui
    )
    target_link_libraries(dna_ui PRIVATE dna_common OpenMP::OpenMP_CXX OpenCL::OpenCL)

    # Copy sw_kernel.cl next to dna_ui so the in-process OpenCL benchmark can find it
    add_custom_command(TARGET dna_ui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/opencl/sw_kernel.cl
            $<TARGET_FILE_DIR:dna_ui>
        COMMENT "Copying sw_kernel.cl for dna_ui OpenCL benchmark"
    )

    if(WIN32)
        # GLFW pre-built on Windows — set GLFW_ROOT or put in external/glfw
        find_package(glfw3 QUIET)
        if(NOT glfw3_FOUND)
            # FetchContent fallback for GLFW on Windows only
            include(FetchContent)
            FetchContent_Declare(
                glfw
                GIT_REPOSITORY https://github.com/glfw/glfw.git
                GIT_TAG        3.4
            )
            set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
            set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
            set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
            FetchContent_MakeAvailable(glfw)
        endif()
        target_link_libraries(dna_ui PRIVATE glfw opengl32 gdi32 user32)
    else()
        # WSL2 / Linux — libglfw3-dev + Mesa OpenGL
        find_package(glfw3 REQUIRED)
        find_package(OpenGL REQUIRED)
        target_link_libraries(dna_ui PRIVATE glfw OpenGL::GL)
    endif()
endif()
