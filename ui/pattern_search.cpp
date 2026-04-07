#include "pattern_search.h"
#include "imgui.h"
#include <cstring>
#include <chrono>
#include <algorithm>
#include <set>

// ── Helper: Highlight text with matches ───────────────────────────────────

static void render_highlighted_text(const char* text, const std::vector<int>& matches, 
                                    int pattern_len, int selected_idx, int hovered_idx,
                                    float highlight_alpha) {
    if (matches.empty() || pattern_len == 0) {
        ImGui::TextWrapped("%s", text);
        return;
    }
    
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 start_pos = ImGui::GetCursorScreenPos();
    int text_len = static_cast<int>(strlen(text));
    
    // Calculate character width using a sample
    ImVec2 char_size = ImGui::CalcTextSize("A");
    float char_width = char_size.x;
    float char_height = char_size.y;
    float line_height = char_height + 6.0f;
    
    // Determine how many characters fit per line
    float available_width = ImGui::GetContentRegionAvail().x - 10.0f;
    int chars_per_line = std::max(1, static_cast<int>(available_width / char_width));
    
    // Draw text character by character with match highlighting
    std::set<int> match_positions;
    for (int pos : matches) {
        for (int i = 0; i < pattern_len && pos + i < text_len; ++i) {
            match_positions.insert(pos + i);
        }
    }
    
    float x = start_pos.x;
    float y = start_pos.y;
    
    for (int i = 0; i < text_len; ++i) {
        // Check if this position is in a match
        bool is_match = match_positions.count(i) > 0;
        
        // Find which match this belongs to
        int match_idx = -1;
        if (is_match) {
            for (size_t m = 0; m < matches.size(); ++m) {
                if (i >= matches[m] && i < matches[m] + pattern_len) {
                    match_idx = static_cast<int>(m);
                    break;
                }
            }
        }
        
        bool is_selected = (match_idx == selected_idx);
        bool is_hovered = (match_idx == hovered_idx);
        
        // Draw background highlight for matches
        if (is_match) {
            ImU32 bg_color;
            if (is_selected) {
                bg_color = IM_COL32(255, 200, 0, 200); // Gold
            } else if (is_hovered) {
                bg_color = IM_COL32(255, 150, 50, 180); // Orange
            } else {
                bg_color = IM_COL32(100, 220, 100, 150); // Green
            }
            
            dl->AddRectFilled(
                ImVec2(x, y - 2),
                ImVec2(x + char_width, y + char_height + 2),
                bg_color,
                2.0f
            );
        }
        
        // Draw character
        char ch[2] = {text[i], '\0'};
        ImU32 text_color = is_match ? IM_COL32(255, 255, 255, 255) : IM_COL32(200, 200, 200, 255);
        dl->AddText(ImVec2(x, y), text_color, ch);
        
        x += char_width;
        
        // Wrap to next line if needed
        if ((i + 1) % chars_per_line == 0 && i + 1 < text_len) {
            x = start_pos.x;
            y += line_height;
        }
    }
    
    // Calculate total height needed
    int num_lines = (text_len + chars_per_line - 1) / chars_per_line;
    ImGui::Dummy(ImVec2(available_width, num_lines * line_height));
}

// ── Helper: Run search and measure performance ────────────────────────────

static PatternSearchResult run_search(PatternAlgorithm algo, 
                                      const std::string& text, 
                                      const std::string& pattern) {
    PatternSearchResult result;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (algo == PatternAlgorithm::KMP) {
        result.positions = kmp_search(text, pattern);
    } else {
        result.positions = boyer_moore_search(text, pattern);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.computed = true;
    
    // Note: Comparison counting would require instrumented versions of the algorithms
    // For now, we estimate based on algorithm complexity
    int n = static_cast<int>(text.size());
    int m = static_cast<int>(pattern.size());
    
    if (algo == PatternAlgorithm::KMP) {
        result.comparisons = n + m; // O(n+m) guaranteed
    } else {
        result.comparisons = n; // Average case for Boyer-Moore
    }
    
    return result;
}

// ── Main render function ───────────────────────────────────────────────────

void render_pattern_search(PatternSearchState& state) {
    // ── Input section ──────────────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.9f, 1.0f, 1.0f));
    ImGui::SeparatorText(" Input");
    ImGui::PopStyleColor();
    
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 1.0f, 1.0f));
    ImGui::Text(" Text (Reference Sequence):");
    ImGui::PopStyleColor();
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputTextMultiline("##text", state.text_buf, sizeof(state.text_buf), 
                                   ImVec2(-1, 90))) {
        // Reset results when text changes
        state.kmp_result.computed = false;
        state.bm_result.computed = false;
    }
    
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.9f, 0.6f, 1.0f));
    ImGui::Text(" Pattern to Search:");
    ImGui::PopStyleColor();
    ImGui::SetNextItemWidth(450.0f);
    if (ImGui::InputText("##pattern", state.pattern_buf, sizeof(state.pattern_buf))) {
        state.kmp_result.computed = false;
        state.bm_result.computed = false;
    }
    
    // ── Algorithm selection ────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 1.0f, 0.6f, 1.0f));
    ImGui::SeparatorText(" Algorithm Selection");
    ImGui::PopStyleColor();
    
    // KMP Radio Button
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.9f, 1.0f, 1.0f));
    if (ImGui::RadioButton(" KMP (Knuth-Morris-Pratt) - O(n+m) guaranteed, best for repeating patterns", 
                          state.algorithm == PatternAlgorithm::KMP)) {
        state.algorithm = PatternAlgorithm::KMP;
    }
    ImGui::PopStyleColor();
    
    // Boyer-Moore Radio Button
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.4f, 1.0f));
    if (ImGui::RadioButton(" Boyer-Moore - O(n/m) average, fast for long patterns in DNA", 
                          state.algorithm == PatternAlgorithm::BoyerMoore)) {
        state.algorithm = PatternAlgorithm::BoyerMoore;
    }
    ImGui::PopStyleColor();
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    
    // Options
    ImGui::Checkbox(" Compare both algorithms", &state.show_comparison);
    ImGui::SameLine();
    ImGui::Checkbox(" Highlight matches", &state.highlight_matches);
    ImGui::SameLine();
    ImGui::Checkbox(" Show metrics", &state.show_metrics);
    
    // ── Search button ──────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.6f, 0.9f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
    if (ImGui::Button(" Search", ImVec2(140, 32))) {
        std::string text(state.text_buf);
        std::string pattern(state.pattern_buf);
        
        if (!pattern.empty() && !text.empty()) {
            if (state.show_comparison) {
                state.kmp_result = run_search(PatternAlgorithm::KMP, text, pattern);
                state.bm_result = run_search(PatternAlgorithm::BoyerMoore, text, pattern);
            } else {
                if (state.algorithm == PatternAlgorithm::KMP) {
                    state.kmp_result = run_search(PatternAlgorithm::KMP, text, pattern);
                    state.bm_result.computed = false;
                } else {
                    state.bm_result = run_search(PatternAlgorithm::BoyerMoore, text, pattern);
                    state.kmp_result.computed = false;
                }
            }
            state.selected_match_idx = -1;
        }
    }
    ImGui::PopStyleColor(3);
    
    // Validation
    if (strlen(state.pattern_buf) == 0) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "   Pattern cannot be empty");
    }
    
    // ── Results section ────────────────────────────────────────────────────
    PatternSearchResult* current_result = nullptr;
    if (state.show_comparison) {
        current_result = (state.kmp_result.computed && state.bm_result.computed) ? 
                        &state.kmp_result : nullptr;
    } else {
        current_result = (state.algorithm == PatternAlgorithm::KMP) ? 
                        (state.kmp_result.computed ? &state.kmp_result : nullptr) :
                        (state.bm_result.computed ? &state.bm_result : nullptr);
    }
    
    if (current_result && current_result->computed) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 1.0f, 0.6f, 1.0f));
        ImGui::SeparatorText(" Results");
        ImGui::PopStyleColor();
        
        // ── Performance metrics ────────────────────────────────────────────
        if (state.show_metrics) {
            if (state.show_comparison && state.kmp_result.computed && state.bm_result.computed) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.9f, 0.5f, 1.0f));
                ImGui::Text(" Algorithm Comparison:");
                ImGui::PopStyleColor();
                ImGui::Spacing();
                
                ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0.2f, 0.3f, 0.4f, 1.0f));
                if (ImGui::BeginTable("comparison", 4, 
                                     ImGuiTableFlags_Borders | 
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_Resizable)) {
                    ImGui::TableSetupColumn(" Algorithm", ImGuiTableColumnFlags_WidthFixed, 120);
                    ImGui::TableSetupColumn(" Time (ms)", ImGuiTableColumnFlags_WidthFixed, 100);
                    ImGui::TableSetupColumn(" Matches", ImGuiTableColumnFlags_WidthFixed, 90);
                    ImGui::TableSetupColumn(" Speedup", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableHeadersRow();
                    
                    // KMP row
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextColored(ImVec4(0.6f, 0.9f, 1.0f, 1.0f), "⚡ KMP");
                    ImGui::TableNextColumn();
                    ImGui::Text("%.3f", state.kmp_result.time_ms);
                    ImGui::TableNextColumn();
                    ImGui::Text("%zu", state.kmp_result.positions.size());
                    ImGui::TableNextColumn();
                    double kmp_speedup = state.bm_result.time_ms / std::max(state.kmp_result.time_ms, 0.001);
                    if (kmp_speedup > 1.1)
                        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%.2fx faster", kmp_speedup);
                    else if (kmp_speedup < 0.9)
                        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "%.2fx slower", 1.0/kmp_speedup);
                    else
                        ImGui::Text("≈ same (%.2fx)", kmp_speedup);
                    
                    // Boyer-Moore row
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), " Boyer-Moore");
                    ImGui::TableNextColumn();
                    ImGui::TableNextColumn();
                    ImGui::Text("%.3f", state.bm_result.time_ms);
                    ImGui::TableNextColumn();
                    ImGui::Text("%zu", state.bm_result.positions.size());
                    ImGui::TableNextColumn();
                    double bm_speedup = state.kmp_result.time_ms / std::max(state.bm_result.time_ms, 0.001);
                    if (bm_speedup > 1.1)
                        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%.2fx faster", bm_speedup);
                    else if (bm_speedup < 0.9)
                        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "%.2fx slower", 1.0/bm_speedup);
                    else
                        ImGui::Text("≈ same (%.2fx)", bm_speedup);
                    
                    ImGui::EndTable();
                }
                ImGui::PopStyleColor();
                ImGui::Spacing();
            } else {
                // Single algorithm results
                const char* algo_name = (state.algorithm == PatternAlgorithm::KMP) ? " KMP" : " Boyer-Moore";
                ImVec4 algo_color = (state.algorithm == PatternAlgorithm::KMP) ? 
                                   ImVec4(0.6f, 0.9f, 1.0f, 1.0f) : ImVec4(1.0f, 0.8f, 0.4f, 1.0f);
                
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.2f, 0.25f, 0.5f));
                ImGui::BeginChild("##metrics", ImVec2(-1, 70), true);
                
                ImGui::Columns(3, "metrics", false);
                
                ImGui::TextColored(algo_color, "%s", algo_name);
                ImGui::Text("Algorithm");
                
                ImGui::NextColumn();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 1.0f, 0.8f, 1.0f));
                ImGui::Text(" %.3f ms", current_result->time_ms);
                ImGui::PopStyleColor();
                ImGui::Text("Execution Time");
                
                ImGui::NextColumn();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.9f, 0.5f, 1.0f));
                ImGui::Text(" %zu matches", current_result->positions.size());
                ImGui::PopStyleColor();
                ImGui::Text("Matches Found");
                
                ImGui::Columns(1);
                ImGui::EndChild();
                ImGui::PopStyleColor();
                
                if (current_result->positions.size() > 0) {
                    int text_len = static_cast<int>(strlen(state.text_buf));
                    int pattern_len = static_cast<int>(strlen(state.pattern_buf));
                    float density = (100.0f * current_result->positions.size() * pattern_len) / 
                                   std::max(text_len, 1);
                    ImGui::Text(" Match Density: %.2f%%", density);
                }
            }
            ImGui::Spacing();
        }
        
        // ── Match positions table ──────────────────────────────────────────
        if (!current_result->positions.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.8f, 1.0f, 1.0f));
            ImGui::SeparatorText(" Match Positions");
            ImGui::PopStyleColor();
            
            ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0.25f, 0.3f, 0.35f, 1.0f));
            if (ImGui::BeginTable("matches", 3, 
                                 ImGuiTableFlags_Borders | 
                                 ImGuiTableFlags_RowBg | 
                                 ImGuiTableFlags_ScrollY,
                                 ImVec2(0, 220))) {
                ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 50);
                ImGui::TableSetupColumn("Position", ImGuiTableColumnFlags_WidthFixed, 90);
                ImGui::TableSetupColumn("Context", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableHeadersRow();
                
                int pattern_len = static_cast<int>(strlen(state.pattern_buf));
                int text_len = static_cast<int>(strlen(state.text_buf));
                
                for (size_t i = 0; i < current_result->positions.size() && i < 100; ++i) {
                    int pos = current_result->positions[i];
                    
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    
                    bool is_selected = (static_cast<int>(i) == state.selected_match_idx);
                    if (ImGui::Selectable(("##match" + std::to_string(i)).c_str(), 
                                         is_selected, ImGuiSelectableFlags_SpanAllColumns)) {
                        state.selected_match_idx = static_cast<int>(i);
                    }
                    
                    if (ImGui::IsItemHovered()) {
                        state.hovered_match_idx = static_cast<int>(i);
                    }
                    
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.9f, 1.0f, 1.0f));
                    ImGui::Text("%zu", i + 1);
                    ImGui::PopStyleColor();
                    
                    ImGui::TableNextColumn();
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.9f, 0.6f, 1.0f));
                    ImGui::Text("%d", pos);
                    ImGui::PopStyleColor();
                    
                    ImGui::TableNextColumn();
                    // Show context: 10 chars before and after
                    int context_start = std::max(0, pos - 10);
                    int context_end = std::min(text_len, pos + pattern_len + 10);
                    
                    std::string prefix = (context_start > 0) ? "..." : "";
                    std::string suffix = (context_end < text_len) ? "..." : "";
                    
                    std::string context = prefix + 
                                         std::string(state.text_buf + context_start, 
                                                    context_end - context_start) + 
                                         suffix;
                    ImGui::TextWrapped("%s", context.c_str());
                }
                
                if (current_result->positions.size() > 100) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("... and %zu more matches", 
                                       current_result->positions.size() - 100);
                }
                
                ImGui::EndTable();
            }
            ImGui::PopStyleColor();
            
            // ── Highlighted text display ───────────────────────────────────
            if (state.highlight_matches) {
                ImGui::Spacing();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.9f, 1.0f));
                ImGui::SeparatorText(" Visual Match Display");
                ImGui::PopStyleColor();
                
                // Visual diagram showing matches
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
                float canvas_width = ImGui::GetContentRegionAvail().x;
                
                // Draw a visual representation
                int text_len = static_cast<int>(strlen(state.text_buf));
                int pattern_len = static_cast<int>(strlen(state.pattern_buf));
                
                if (text_len > 0 && pattern_len > 0) {
                    float bar_width = std::min(canvas_width - 20.0f, text_len * 8.0f);
                    float char_width = bar_width / text_len;
                    float bar_height = 25.0f;
                    
                    // Draw text bar
                    dl->AddRectFilled(
                        ImVec2(canvas_pos.x, canvas_pos.y),
                        ImVec2(canvas_pos.x + bar_width, canvas_pos.y + bar_height),
                        IM_COL32(50, 60, 70, 255),
                        3.0f
                    );
                    
                    dl->AddText(ImVec2(canvas_pos.x + 5, canvas_pos.y + 5), 
                               IM_COL32(150, 150, 150, 255), "Text Sequence");
                    
                    // Draw match highlights on the bar
                    for (size_t m = 0; m < current_result->positions.size() && m < 50; ++m) {
                        int pos = current_result->positions[m];
                        float x_start = canvas_pos.x + (pos * char_width);
                        float x_end = canvas_pos.x + ((pos + pattern_len) * char_width);
                        
                        ImU32 match_color;
                        if (static_cast<int>(m) == state.selected_match_idx) {
                            match_color = IM_COL32(255, 200, 0, 230); // Gold
                        } else if (static_cast<int>(m) == state.hovered_match_idx) {
                            match_color = IM_COL32(255, 150, 50, 200); // Orange
                        } else {
                            match_color = IM_COL32(100, 220, 100, 180); // Green
                        }
                        
                        dl->AddRectFilled(
                            ImVec2(x_start, canvas_pos.y),
                            ImVec2(x_end, canvas_pos.y + bar_height),
                            match_color,
                            3.0f
                        );
                        
                        // Add match number
                        if (char_width * pattern_len > 15.0f) {
                            char label[8];
                            std::snprintf(label, sizeof(label), "%zu", m + 1);
                            dl->AddText(ImVec2(x_start + 3, canvas_pos.y + 8),
                                       IM_COL32_WHITE, label);
                        }
                    }
                    
                    ImGui::Dummy(ImVec2(bar_width, bar_height + 10.0f));
                    
                    // Legend
                    ImGui::Spacing();
                    ImGui::Text("Legend:");
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.4f, 0.86f, 0.4f, 1.0f), "■");
                    ImGui::SameLine();
                    ImGui::Text("Match");
                    ImGui::SameLine();
                    ImGui::Spacing();
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.2f, 1.0f), "■");
                    ImGui::SameLine();
                    ImGui::Text("Hovered");
                    ImGui::SameLine();
                    ImGui::Spacing();
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.0f, 1.0f), "■");
                    ImGui::SameLine();
                    ImGui::Text("Selected");
                }
                
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                
                // Character-by-character view
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.14f, 0.16f, 1.0f));
                ImGui::BeginChild("##highlighted_text", ImVec2(0, 180), true);
                render_highlighted_text(state.text_buf, current_result->positions,
                                       static_cast<int>(strlen(state.pattern_buf)),
                                       state.selected_match_idx, state.hovered_match_idx,
                                       state.highlight_alpha);
                ImGui::EndChild();
                ImGui::PopStyleColor();
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.5f, 1.0f));
            ImGui::Text(" No matches found.");
            ImGui::PopStyleColor();
        }
    } else if (!state.kmp_result.computed && !state.bm_result.computed) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.8f, 1.0f));
        ImGui::Text(" Enter a text and pattern, then click Search to find matches.");
        ImGui::PopStyleColor();
    }
    
    // ── Statistics ─────────────────────────────────────────────────────────
    if (current_result && current_result->computed && state.show_metrics) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.9f, 1.0f, 1.0f));
        ImGui::SeparatorText(" Statistics");
        ImGui::PopStyleColor();
        
        ImGui::Columns(3, "stats_cols", false);
        
        ImGui::Text(" Text Length:");
        ImGui::TextColored(ImVec4(0.8f, 1.0f, 0.8f, 1.0f), "%zu characters", strlen(state.text_buf));
        
        ImGui::NextColumn();
        ImGui::Text(" Pattern Length:");
        ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.6f, 1.0f), "%zu characters", strlen(state.pattern_buf));
        
        ImGui::NextColumn();
        ImGui::Text(" Total Matches:");
        ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "%zu", current_result->positions.size());
        
        ImGui::Columns(1);
        
        if (!current_result->positions.empty()) {
            int first_match = current_result->positions.front();
            int last_match = current_result->positions.back();
            
            ImGui::Spacing();
            ImGui::Columns(3, "match_stats", false);
            
            ImGui::Text("📍 First Match:");
            ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "position %d", first_match);
            
            ImGui::NextColumn();
            ImGui::Text("📍 Last Match:");
            ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "position %d", last_match);
            
            ImGui::NextColumn();
            ImGui::Text("📏 Search Span:");
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.7f, 1.0f), "%d characters", last_match - first_match);
            
            ImGui::Columns(1);
        }
    }
}
