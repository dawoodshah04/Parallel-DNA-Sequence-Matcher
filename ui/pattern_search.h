#pragma once

// Pattern search state and rendering

#include "kmp.h"
#include <string>
#include <vector>
#include <chrono>

// ── Pattern Search state ───────────────────────────────────────────────────

enum class PatternAlgorithm {
    KMP,
    BoyerMoore
};

struct PatternSearchResult {
    std::vector<int> positions;
    double time_ms = 0.0;
    int comparisons = 0;
    bool computed = false;
};

struct PatternSearchState {
    char text_buf[2048] = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";
    char pattern_buf[256] = "ACGT";
    
    PatternAlgorithm algorithm = PatternAlgorithm::KMP;
    
    PatternSearchResult kmp_result;
    PatternSearchResult bm_result;
    
    bool show_comparison = false;
    bool highlight_matches = true;
    bool show_metrics = true;
    
    int selected_match_idx = -1;
    int hovered_match_idx = -1;
    
    // Display options
    float highlight_alpha = 0.4f;
    bool case_sensitive = true;
};

// Defined in pattern_search.cpp
void render_pattern_search(PatternSearchState& state);
