#include "smith_waterman.h"

#include <algorithm>

// ── Internal helpers ───────────────────────────────────────────────────────

static inline int base_score(char a, char b) {
    return (a == b) ? SW_MATCH : SW_MISMATCH;
}

// ── DP matrix ──────────────────────────────────────────────────────────────

std::vector<std::vector<int>> smith_waterman_matrix(const std::string& query,
                                                     const std::string& ref) {
    int m = static_cast<int>(query.size());
    int n = static_cast<int>(ref.size());

    // H[i][j]: row i = query position (1-indexed), col j = ref position (1-indexed)
    // Row 0 and column 0 are the zero boundary.
    std::vector<std::vector<int>> H(m + 1, std::vector<int>(n + 1, 0));

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int diag = H[i - 1][j - 1] + base_score(query[i - 1], ref[j - 1]);
            int up   = H[i - 1][j]     + SW_GAP;
            int left = H[i][j - 1]     + SW_GAP;
            H[i][j]  = std::max({0, diag, up, left});
        }
    }

    return H;
}

// ── Traceback helper (shared by smith_waterman and smith_waterman_full) ────

static AlignmentResult traceback(const std::vector<std::vector<int>>& H,
                                  const std::string& query,
                                  const std::string& ref,
                                  int max_i, int max_j, int max_score,
                                  std::vector<std::pair<int,int>>* path_out) {
    int n = static_cast<int>(ref.size());
    std::string aq, ar;
    int i = max_i, j = max_j;

    while (i > 0 && j > 0 && H[i][j] > 0) {
        if (path_out) path_out->push_back({i, j});

        int diag = H[i - 1][j - 1] + base_score(query[i - 1], ref[j - 1]);
        if (H[i][j] == diag) {
            aq += query[i - 1];
            ar += ref[j - 1];
            --i; --j;
        } else if (H[i][j] == H[i - 1][j] + SW_GAP) {
            aq += query[i - 1];
            ar += '-';
            --i;
        } else {
            aq += '-';
            ar += ref[j - 1];
            --j;
        }
    }

    std::reverse(aq.begin(), aq.end());
    std::reverse(ar.begin(), ar.end());

    (void)n;  // suppress unused warning
    return AlignmentResult{max_score, aq, ar, i, j};
}

// ── Public API ─────────────────────────────────────────────────────────────

AlignmentResult smith_waterman(const std::string& query, const std::string& ref) {
    auto H = smith_waterman_matrix(query, ref);

    int m = static_cast<int>(query.size());
    int n = static_cast<int>(ref.size());

    int max_score = 0, max_i = 0, max_j = 0;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (H[i][j] > max_score) {
                max_score = H[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }

    return traceback(H, query, ref, max_i, max_j, max_score, nullptr);
}

SWVisualizationData smith_waterman_full(const std::string& query,
                                        const std::string& ref) {
    SWVisualizationData viz;
    viz.matrix = smith_waterman_matrix(query, ref);

    int m = static_cast<int>(query.size());
    int n = static_cast<int>(ref.size());

    int max_score = 0, max_i = 0, max_j = 0;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (viz.matrix[i][j] > max_score) {
                max_score = viz.matrix[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }

    viz.result = traceback(viz.matrix, query, ref,
                           max_i, max_j, max_score,
                           &viz.traceback_cells);
    return viz;
}
