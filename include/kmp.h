#pragma once

#include <string>
#include <vector>

// ── KMP exact pattern search ───────────────────────────────────────────────

// Build the KMP failure function (prefix table) for the given pattern.
// Returns a vector pi of length pattern.size() where pi[i] is the length
// of the longest proper prefix of pattern[0..i] that is also a suffix.
std::vector<int> build_failure_function(const std::string& pattern);

// Search for all occurrences of pattern in text using KMP.
// Returns the 0-indexed start positions of every match.
std::vector<int> kmp_search(const std::string& text, const std::string& pattern);

// ── Boyer-Moore exact pattern search ──────────────────────────────────────

// Search for all occurrences of pattern in text using Boyer-Moore
// (bad-character + good-suffix heuristics).
// Returns the 0-indexed start positions of every match.
std::vector<int> boyer_moore_search(const std::string& text,
                                    const std::string& pattern);
