#pragma once

#include "dna_utils.h"
#include <vector>
#include <utility>

// ── Scoring constants ──────────────────────────────────────────────────────
constexpr int SW_MATCH    =  2;
constexpr int SW_MISMATCH = -1;
constexpr int SW_GAP      = -2;

// ── Result types ───────────────────────────────────────────────────────────

// Full visualization data returned by smith_waterman_full():
//   matrix          — complete H[m+1][n+1] DP table (row = query, col = ref)
//   result          — alignment score + aligned strings
//   traceback_cells — (row, col) cells visited during traceback, in reverse
//                     order (from max-cell back to start); row/col are 1-indexed
struct SWVisualizationData {
    std::vector<std::vector<int>>    matrix;
    AlignmentResult                  result;
    std::vector<std::pair<int,int>>  traceback_cells;
};

// ── API ────────────────────────────────────────────────────────────────────

// Compute Smith-Waterman local alignment and return only the result.
// All CPU variants (sequential, OpenMP, MPI) call this function.
AlignmentResult smith_waterman(const std::string& query, const std::string& ref);

// Return the full (m+1) x (n+1) DP matrix without traceback.
// Used internally and by the OpenCL variant for verification.
std::vector<std::vector<int>> smith_waterman_matrix(const std::string& query,
                                                     const std::string& ref);

// Return alignment result + full matrix + traceback path in one call.
// Used by the ImGui visualizer.
SWVisualizationData smith_waterman_full(const std::string& query,
                                        const std::string& ref);
