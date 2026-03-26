#include "kmp.h"

#include <algorithm>
#include <unordered_map>

// ══════════════════════════════════════════════════════════════════════════
// KMP
// ══════════════════════════════════════════════════════════════════════════

std::vector<int> build_failure_function(const std::string& pattern) {
    int m = static_cast<int>(pattern.size());
    std::vector<int> pi(m, 0);
    int k = 0;
    for (int i = 1; i < m; ++i) {
        while (k > 0 && pattern[k] != pattern[i]) k = pi[k - 1];
        if (pattern[k] == pattern[i]) ++k;
        pi[i] = k;
    }
    return pi;
}

std::vector<int> kmp_search(const std::string& text, const std::string& pattern) {
    if (pattern.empty()) return {};

    std::vector<int> positions;
    auto pi = build_failure_function(pattern);
    int n = static_cast<int>(text.size());
    int m = static_cast<int>(pattern.size());
    int q = 0;

    for (int i = 0; i < n; ++i) {
        while (q > 0 && pattern[q] != text[i]) q = pi[q - 1];
        if (pattern[q] == text[i]) ++q;
        if (q == m) {
            positions.push_back(i - m + 1);
            q = pi[q - 1];
        }
    }
    return positions;
}

// ══════════════════════════════════════════════════════════════════════════
// Boyer-Moore (bad-character + good-suffix)
// ══════════════════════════════════════════════════════════════════════════

// Bad-character table: last occurrence of each character in pattern
static std::unordered_map<char, int> build_bad_char(const std::string& pattern) {
    std::unordered_map<char, int> bc;
    int m = static_cast<int>(pattern.size());
    for (int i = 0; i < m; ++i) bc[pattern[i]] = i;
    return bc;
}

// Good-suffix shift table (Knuth / Apostolico-Giancarlo variant)
static std::vector<int> build_good_suffix(const std::string& pattern) {
    int m = static_cast<int>(pattern.size());
    std::vector<int> shift(m + 1, m);
    std::vector<int> border(m + 1, 0);

    int i = m, j = m + 1;
    border[i] = j;
    while (i > 0) {
        while (j <= m && pattern[i - 1] != pattern[j - 1]) {
            if (shift[j] == m) shift[j] = j - i;
            j = border[j];
        }
        --i; --j;
        border[i] = j;
    }

    j = border[0];
    for (i = 0; i <= m; ++i) {
        if (shift[i] == m) shift[i] = j;
        if (i == j)        j = border[j];
    }

    return shift;
}

std::vector<int> boyer_moore_search(const std::string& text,
                                    const std::string& pattern) {
    if (pattern.empty()) return {};

    std::vector<int> positions;
    int n = static_cast<int>(text.size());
    int m = static_cast<int>(pattern.size());

    auto bc = build_bad_char(pattern);
    auto gs = build_good_suffix(pattern);

    int s = 0;
    while (s <= n - m) {
        int j = m - 1;
        while (j >= 0 && pattern[j] == text[s + j]) --j;

        if (j < 0) {
            positions.push_back(s);
            s += gs[0];
        } else {
            int bc_val = bc.count(text[s + j]) ? bc.at(text[s + j]) : -1;
            int bc_shift = j - bc_val;
            s += std::max(gs[j + 1], bc_shift);
        }
    }
    return positions;
}
