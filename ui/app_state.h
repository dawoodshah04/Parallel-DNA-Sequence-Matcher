#pragma once

// Private header — only used within src/ui/*.cpp

#include "dna_utils.h"
#include "smith_waterman.h"
#include "kmp.h"

#include <string>
#include <vector>

// ── SW Visualizer state ────────────────────────────────────────────────────

struct SWVisualizerState {
    char query_buf[128] = "ACGTACGTGCATGCAT";
    char ref_buf[128]   = "TGCATGCATACGTACG";

    SWVisualizationData viz;
    bool computed = false;
    float cell_size = 22.0f;    // pixels per DP cell
    int   max_vis   = 40;       // maximum rows/cols to render in grid
    
    // Animation state
    bool animating = false;
    bool paused = false;
    int current_step = 0;       // current animation step (index into computation_steps)
    float animation_speed = 50.0f; // steps per second
    float step_accumulator = 0.0f;
    
    // Scoring parameters
    ScoringParams scoring;
    
    // Interaction state
    int hovered_row = -1;
    int hovered_col = -1;
    int selected_row = -1;
    int selected_col = -1;
    
    // Display options
    bool show_statistics = true;
    bool show_cell_values = true;
};

// Defined in sw_visualizer.cpp
void render_sw_visualizer(SWVisualizerState& state);

// ── Benchmark panel state ──────────────────────────────────────────────────

struct BenchmarkEntry {
    const char* label;
    double      time_ms  = 0.0;
    int         score    = 0;
    bool        measured = false;
    std::string error;          // non-empty if the last run failed
};

struct BenchmarkState {
    int seq_length = 300;       // length of random test sequences
    int db_size    = 20;        // number of DB sequences
    int mpi_ranks  = 4;         // MPI ranks to launch via mpiexec

    std::vector<BenchmarkEntry> entries = {
        {"Sequential", 0.0, 0, false, ""},
        {"OpenMP 2T",  0.0, 0, false, ""},
        {"OpenMP 4T",  0.0, 0, false, ""},
        {"OpenMP 8T",  0.0, 0, false, ""},
        {"MPI",        0.0, 0, false, ""},
        {"OpenCL",     0.0, 0, false, ""},
    };

    bool running = false;
};

// Defined in benchmark_panel.cpp
void render_benchmark_panel(BenchmarkState& state);
