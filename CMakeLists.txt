cmake_minimum_required(VERSION 3.20)
project(ParallelDNASequenceMatcher CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Common static library ──────────────────────────────────────────────────
add_library(dna_common STATIC
    src/common/dna_utils.cpp
    src/common/smith_waterman.cpp
    src/common/kmp.cpp
)
target_include_directories(dna_common PUBLIC include)

# ── Sequential baseline ────────────────────────────────────────────────────
add_executable(dna_sequential src/sequential/main_sequential.cpp)
target_link_libraries(dna_sequential PRIVATE dna_common)

# ── OpenMP ─────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)
add_executable(dna_openmp src/openmp/main_openmp.cpp)
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
        add_executable(dna_mpi src/mpi/main_mpi.cpp)
        target_include_directories(dna_mpi PRIVATE ${MSMPI_INCLUDE_DIR})
        target_link_libraries(dna_mpi PRIVATE dna_common ${MSMPI_LIB})
    endif()
else()
    # WSL2 / Linux — OpenMPI
    find_package(MPI REQUIRED)
    add_executable(dna_mpi src/mpi/main_mpi.cpp)
    target_link_libraries(dna_mpi PRIVATE dna_common MPI::MPI_CXX)
endif()

# ── OpenCL ─────────────────────────────────────────────────────────────────
find_package(OpenCL REQUIRED)
add_executable(dna_opencl src/opencl/main_opencl.cpp)
target_link_libraries(dna_opencl PRIVATE dna_common OpenCL::OpenCL)

# Copy .cl kernel files next to the executable after build
add_custom_command(TARGET dna_opencl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/src/opencl/sw_kernel.cl
        ${CMAKE_SOURCE_DIR}/src/opencl/kmp_kernel.cl
        $<TARGET_FILE_DIR:dna_opencl>
    COMMENT "Copying OpenCL kernel files to build directory"
)

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
        src/ui/main_ui.cpp
        src/ui/sw_visualizer.cpp
        src/ui/benchmark_panel.cpp
        ${IMGUI_SOURCES}
    )
    target_include_directories(dna_ui PRIVATE
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        include
        src/ui
    )
    target_link_libraries(dna_ui PRIVATE dna_common OpenMP::OpenMP_CXX OpenCL::OpenCL)

    # Copy sw_kernel.cl next to dna_ui so the in-process OpenCL benchmark can find it
    add_custom_command(TARGET dna_ui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/src/opencl/sw_kernel.cl
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
