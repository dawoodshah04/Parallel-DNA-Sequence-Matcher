#pragma once

#include "dna_utils.h"
#include <vector>
#include <utility>

// ── Scoring constants ──────────────────────────────────────────────────────
constexpr int SW_MATCH    =  2;
constexpr int SW_MISMATCH = -1;
constexpr int SW_GAP      = -2;

// Configurable scoring parameters for UI
struct ScoringParams {
    int match    = SW_MATCH;
    int mismatch = SW_MISMATCH;
    int gap      = SW_GAP;
};

// ── Result types ───────────────────────────────────────────────────────────

// Cell computation details for step-by-step visualization
struct CellComputation {
    int row, col;           // position in matrix
    int score;              // final score for this cell
    int from_left;          // score if came from left (gap in query)
    int from_top;           // score if came from top (gap in ref)
    int from_diag;          // score if came from diagonal (match/mismatch)
    int chosen_direction;   // 0=zero, 1=diag, 2=left, 3=top
    bool is_match;          // true if query[row-1] == ref[col-1]
};

// Full visualization data returned by smith_waterman_full():
//   matrix          — complete H[m+1][n+1] DP table (row = query, col = ref)
//   result          — alignment score + aligned strings
//   traceback_cells — (row, col) cells visited during traceback, in reverse
//                     order (from max-cell back to start); row/col are 1-indexed
//   computation_steps — step-by-step cell computations (row-major order)
struct SWVisualizationData {
    std::vector<std::vector<int>>    matrix;
    AlignmentResult                  result;
    std::vector<std::pair<int,int>>  traceback_cells;
    std::vector<CellComputation>     computation_steps;
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

// Return full visualization data with custom scoring parameters.
SWVisualizationData smith_waterman_full(const std::string& query,
                                        const std::string& ref,
                                        const ScoringParams& params);
