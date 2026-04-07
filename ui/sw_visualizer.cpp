#include "app_state.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <set>

#include "imgui.h"

// ── Color mapping for DP cell values ──────────────────────────────────────

static ImU32 cell_color(int val, int max_val, bool on_traceback, bool is_computing = false, 
                       bool is_pending = false, bool is_hovered = false, bool is_selected = false) {
    if (is_selected)
        return IM_COL32(255, 200, 0, 255);   // gold - selected cell
    if (is_hovered)
        return IM_COL32(255, 160, 80, 255);  // orange - hovered cell
    if (is_computing)
        return IM_COL32(100, 160, 255, 255); // blue - currently computing
    if (is_pending)
        return IM_COL32(60, 60, 70, 255);    // dark gray - not computed yet
    if (on_traceback)
        return IM_COL32(220, 60, 60, 230);   // red — traceback path
    if (val == 0)
        return IM_COL32(40, 40, 45, 255);    // near-black — zero cells
    float t = (max_val > 0) ? static_cast<float>(val) / static_cast<float>(max_val) : 0.0f;
    // Green gradient: dark green → bright green
    auto r = static_cast<int>(20  + 30  * t);
    auto g = static_cast<int>(80  + 175 * t);
    auto b = static_cast<int>(20  + 20  * t);
    return IM_COL32(r, g, b, 255);
}

// ── Alignment display helper ───────────────────────────────────────────────

static void show_alignment(const AlignmentResult& res) {
    if (res.aligned_query.empty()) {
        ImGui::TextDisabled("(no alignment)");
        return;
    }
    const auto& aq = res.aligned_query;
    const auto& ar = res.aligned_ref;
    std::size_t len = std::min(aq.size(), ar.size());

    // Build match-indicator string
    std::string mid(len, ' ');
    int matches = 0, mismatches = 0, gaps = 0;
    for (std::size_t k = 0; k < len; ++k) {
        if (aq[k] == '-' || ar[k] == '-') {
            gaps++;
        } else if (aq[k] == ar[k]) {
            mid[k] = '|';
            matches++;
        } else {
            mismatches++;
        }
    }

    ImGui::Text("Query : %s", aq.c_str());
    ImGui::Text("        %s", mid.c_str());
    ImGui::Text("Ref   : %s", ar.c_str());
    ImGui::Spacing();
    ImGui::Text("Score: %d  |  Matches: %d  |  Mismatches: %d  |  Gaps: %d",
                res.score, matches, mismatches, gaps);
    
    // Calculate statistics
    int total_aligned = matches + mismatches;
    if (total_aligned > 0) {
        float identity = (100.0f * matches) / total_aligned;
        ImGui::Text("Identity: %.1f%%  |  Length: %zu  |  Query start: %d  |  Ref start: %d",
                    identity, len, res.query_start, res.ref_start);
    }
}

// ── Main render function ───────────────────────────────────────────────────

void render_sw_visualizer(SWVisualizerState& state) {
    // ── Input section ──────────────────────────────────────────────────────
    ImGui::SeparatorText("Input Sequences");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
    ImGui::InputText("Query##sw", state.query_buf, sizeof(state.query_buf));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::InputText("Reference##sw", state.ref_buf, sizeof(state.ref_buf));

    // ── Scoring parameters ─────────────────────────────────────────────────
    ImGui::SeparatorText("Scoring Parameters");
    ImGui::SetNextItemWidth(150);
    bool scoring_changed = ImGui::SliderInt("Match", &state.scoring.match, 1, 5);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    scoring_changed |= ImGui::SliderInt("Mismatch", &state.scoring.mismatch, -5, -1);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    scoring_changed |= ImGui::SliderInt("Gap", &state.scoring.gap, -5, -1);
    
    if (scoring_changed && state.computed && !state.animating) {
        // Recalculate with new parameters
        state.viz = smith_waterman_full(state.query_buf, state.ref_buf, state.scoring);
        state.current_step = static_cast<int>(state.viz.computation_steps.size());
    }

    // ── Visualization controls ─────────────────────────────────────────────
    ImGui::SeparatorText("Visualization Controls");
    ImGui::SliderFloat("Cell size (px)", &state.cell_size, 10.0f, 50.0f);
    ImGui::SameLine();
    ImGui::SliderInt("Max visible cells", &state.max_vis, 10, 80);
    
    ImGui::Checkbox("Show cell values", &state.show_cell_values);
    ImGui::SameLine();
    ImGui::Checkbox("Show statistics", &state.show_statistics);
    ImGui::Spacing();

    // ── Run button ─────────────────────────────────────────────────────────
    bool run_clicked = ImGui::Button("Compute Alignment", ImVec2(180, 0));
    if (run_clicked && std::strlen(state.query_buf) > 0 &&
                       std::strlen(state.ref_buf)   > 0) {
        state.viz = smith_waterman_full(state.query_buf, state.ref_buf, state.scoring);
        state.computed = true;
        state.animating = false;
        state.paused = false;
        state.current_step = static_cast<int>(state.viz.computation_steps.size());
    }

    if (!state.computed) {
        ImGui::TextDisabled("Enter sequences above and press Compute Alignment.");
        return;
    }

    // ── Animation controls ─────────────────────────────────────────────────
    ImGui::SameLine();
    if (ImGui::Button(state.animating && !state.paused ? "Pause" : "Play Animation", ImVec2(140, 0))) {
        if (!state.animating) {
            state.animating = true;
            state.paused = false;
            state.current_step = 0;
            state.step_accumulator = 0.0f;
        } else {
            state.paused = !state.paused;
        }
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Reset", ImVec2(80, 0))) {
        state.animating = false;
        state.paused = false;
        state.current_step = static_cast<int>(state.viz.computation_steps.size());
    }
    
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    ImGui::SliderFloat("Speed (steps/sec)", &state.animation_speed, 5.0f, 200.0f, "%.0f");

    // ── Animation update ───────────────────────────────────────────────────
    if (state.animating && !state.paused) {
        float dt = ImGui::GetIO().DeltaTime;
        state.step_accumulator += dt * state.animation_speed;
        
        int steps_to_advance = static_cast<int>(state.step_accumulator);
        state.step_accumulator -= steps_to_advance;
        
        state.current_step += steps_to_advance;
        if (state.current_step >= static_cast<int>(state.viz.computation_steps.size())) {
            state.current_step = static_cast<int>(state.viz.computation_steps.size());
            state.animating = false;
            state.paused = false;
        }
    }

    // Progress bar for animation
    if (!state.viz.computation_steps.empty()) {
        float progress = static_cast<float>(state.current_step) / state.viz.computation_steps.size();
        ImGui::ProgressBar(progress, ImVec2(-1, 0), 
                          (state.animating ? "Computing..." : "Complete"));
        ImGui::Text("Step %d / %zu", state.current_step, state.viz.computation_steps.size());
    }

    // ── Alignment result ───────────────────────────────────────────────────
    if (state.current_step >= static_cast<int>(state.viz.computation_steps.size())) {
        ImGui::Spacing();
        ImGui::SeparatorText("Alignment Result");
        show_alignment(state.viz.result);
    }

    // ── DP matrix heatmap ─────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::SeparatorText("DP Matrix Visualization");

    const auto& H    = state.viz.matrix;
    int m_full       = static_cast<int>(H.size()) - 1;       // query length
    int n_full       = static_cast<int>(H[0].size()) - 1;    // ref length
    int m_vis        = std::min(m_full, state.max_vis);
    int n_vis        = std::min(n_full, state.max_vis);
    float cell_sz    = state.cell_size;

    // Find global max for colour scaling
    int max_val = 1;
    for (int i = 0; i <= m_vis; ++i)
        for (int j = 0; j <= n_vis; ++j)
            if (H[i][j] > max_val) max_val = H[i][j];

    // Build traceback set for fast lookup
    std::set<std::pair<int,int>> on_tb_set;
    if (state.current_step >= static_cast<int>(state.viz.computation_steps.size())) {
        for (auto& [pi, pj] : state.viz.traceback_cells) {
            if (pi <= m_vis && pj <= n_vis)
                on_tb_set.insert({pi, pj});
        }
    }

    // Build set of computed cells based on animation state
    std::set<std::pair<int,int>> computed_cells;
    int computing_row = -1, computing_col = -1;
    
    for (int s = 0; s < state.current_step && s < static_cast<int>(state.viz.computation_steps.size()); ++s) {
        const auto& step = state.viz.computation_steps[s];
        computed_cells.insert({step.row, step.col});
    }
    
    if (state.animating && state.current_step > 0 && 
        state.current_step <= static_cast<int>(state.viz.computation_steps.size())) {
        const auto& current = state.viz.computation_steps[state.current_step - 1];
        computing_row = current.row;
        computing_col = current.col;
    }

    ImVec2 canvas = ImGui::GetCursorScreenPos();
    float total_w = (n_vis + 2) * cell_sz;
    float total_h = (m_vis + 2) * cell_sz;

    // Get mouse position for hover detection
    ImVec2 mouse_pos = ImGui::GetMousePos();
    bool mouse_in_grid = ImGui::IsMouseHoveringRect(canvas, ImVec2(canvas.x + total_w, canvas.y + total_h));
    
    if (mouse_in_grid) {
        int hover_j = static_cast<int>((mouse_pos.x - canvas.x) / cell_sz);
        int hover_i = static_cast<int>((mouse_pos.y - canvas.y) / cell_sz);
        
        if (hover_i >= 0 && hover_i <= m_vis && hover_j >= 0 && hover_j <= n_vis) {
            state.hovered_row = hover_i;
            state.hovered_col = hover_j;
            
            // Click to select
            if (ImGui::IsMouseClicked(0)) {
                state.selected_row = hover_i;
                state.selected_col = hover_j;
            }
        } else {
            state.hovered_row = -1;
            state.hovered_col = -1;
        }
    } else {
        state.hovered_row = -1;
        state.hovered_col = -1;
    }

    // Reserve space so ImGui scrollbar appears when needed
    ImGui::InvisibleButton("##matrix_area", ImVec2(total_w + 4, total_h + 4));

    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Draw cells
    for (int i = 0; i <= m_vis; ++i) {
        for (int j = 0; j <= n_vis; ++j) {
            float x0 = canvas.x + j * cell_sz;
            float y0 = canvas.y + i * cell_sz;
            float x1 = x0 + cell_sz - 1;
            float y1 = y0 + cell_sz - 1;

            bool on_tb = on_tb_set.count({i, j}) > 0;
            bool is_computed = computed_cells.count({i, j}) > 0 || 
                             state.current_step >= static_cast<int>(state.viz.computation_steps.size());
            bool is_computing = (i == computing_row && j == computing_col);
            bool is_pending = !is_computed && !is_computing && (i > 0 && j > 0);
            bool is_hovered = (i == state.hovered_row && j == state.hovered_col);
            bool is_selected = (i == state.selected_row && j == state.selected_col);
            
            int cell_value = is_computed ? H[i][j] : 0;
            
            ImU32 col = cell_color(cell_value, max_val, on_tb, is_computing, 
                                  is_pending, is_hovered, is_selected);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), col);
            dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(80, 80, 80, 255));

            // Show value text when cells are large enough and computed
            if (state.show_cell_values && cell_sz >= 18 && is_computed) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "%d", cell_value);
                dl->AddText(ImVec2(x0 + 3, y0 + 2), IM_COL32_WHITE, buf);
            }
        }
    }

    // Row and column labels
    for (int j = 1; j <= n_vis; ++j) {
        float x = canvas.x + j * cell_sz + cell_sz * 0.3f;
        float y = canvas.y - cell_sz * 0.7f;
        char buf[4];
        std::snprintf(buf, sizeof(buf), "%c",
            state.ref_buf[static_cast<std::size_t>(j - 1)]);
        dl->AddText(ImVec2(x, y), IM_COL32(200, 200, 200, 255), buf);
    }
    for (int i = 1; i <= m_vis; ++i) {
        float x = canvas.x - cell_sz * 0.5f;
        float y = canvas.y + i * cell_sz + cell_sz * 0.2f;
        char buf[4];
        std::snprintf(buf, sizeof(buf), "%c",
            state.query_buf[static_cast<std::size_t>(i - 1)]);
        dl->AddText(ImVec2(x, y), IM_COL32(200, 200, 200, 255), buf);
    }

    // ── Cell tooltip and details ───────────────────────────────────────────
    if (state.hovered_row >= 0 && state.hovered_col >= 0) {
        ImGui::BeginTooltip();
        ImGui::Text("Cell [%d, %d]", state.hovered_row, state.hovered_col);
        
        if (state.hovered_row == 0 || state.hovered_col == 0) {
            ImGui::Text("Boundary cell (value = 0)");
        } else if (state.current_step >= static_cast<int>(state.viz.computation_steps.size())) {
            // Find the computation step for this cell
            for (const auto& step : state.viz.computation_steps) {
                if (step.row == state.hovered_row && step.col == state.hovered_col) {
                    char q_base = state.query_buf[state.hovered_row - 1];
                    char r_base = state.ref_buf[state.hovered_col - 1];
                    
                    ImGui::Separator();
                    ImGui::Text("Query[%d] = %c, Ref[%d] = %c %s", 
                               state.hovered_row - 1, q_base,
                               state.hovered_col - 1, r_base,
                               step.is_match ? "(MATCH)" : "(mismatch)");
                    ImGui::Separator();
                    ImGui::Text("From diagonal: %d", step.from_diag);
                    ImGui::Text("From left (gap in query): %d", step.from_left);
                    ImGui::Text("From top (gap in ref): %d", step.from_top);
                    ImGui::Separator();
                    
                    const char* dir_name[] = {"Zero", "Diagonal", "Left", "Top"};
                    ImGui::Text("Chosen: %s → Score = %d", 
                               dir_name[step.chosen_direction], step.score);
                    break;
                }
            }
        }
        ImGui::EndTooltip();
    }

    // Show selected cell details below matrix
    if (state.selected_row > 0 && state.selected_col > 0 && 
        state.current_step >= static_cast<int>(state.viz.computation_steps.size())) {
        ImGui::Spacing();
        ImGui::SeparatorText("Selected Cell Details");
        
        for (const auto& step : state.viz.computation_steps) {
            if (step.row == state.selected_row && step.col == state.selected_col) {
                char q_base = state.query_buf[state.selected_row - 1];
                char r_base = state.ref_buf[state.selected_col - 1];
                
                ImGui::Text("Position: [%d, %d]  |  Query[%d] = %c  |  Ref[%d] = %c",
                           state.selected_row, state.selected_col,
                           state.selected_row - 1, q_base,
                           state.selected_col - 1, r_base);
                
                ImGui::Text("Match: %s  |  Final Score: %d",
                           step.is_match ? "YES" : "NO", step.score);
                
                ImGui::Spacing();
                ImGui::Columns(2);
                ImGui::Text("Source"); ImGui::NextColumn();
                ImGui::Text("Score"); ImGui::NextColumn();
                ImGui::Separator();
                
                ImGui::Text("Diagonal [%d,%d]", state.selected_row-1, state.selected_col-1); 
                ImGui::NextColumn();
                ImGui::Text("%d", step.from_diag); 
                ImGui::NextColumn();
                
                ImGui::Text("Left [%d,%d]", state.selected_row, state.selected_col-1); 
                ImGui::NextColumn();
                ImGui::Text("%d", step.from_left); 
                ImGui::NextColumn();
                
                ImGui::Text("Top [%d,%d]", state.selected_row-1, state.selected_col); 
                ImGui::NextColumn();
                ImGui::Text("%d", step.from_top); 
                ImGui::NextColumn();
                
                ImGui::Columns(1);
                
                const char* dir_name[] = {"Zero (reset)", "Diagonal (align)", 
                                         "Left (gap in query)", "Top (gap in ref)"};
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), 
                                  "→ Chosen: %s", dir_name[step.chosen_direction]);
                break;
            }
        }
    }

    if (m_vis < m_full || n_vis < n_full) {
        ImGui::Spacing();
        ImGui::TextDisabled(
            "(showing %dx%d of %dx%d — increase 'Max visible cells' to see more)",
            m_vis, n_vis, m_full, n_full);
    }

    // ── Legend ─────────────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.86f, 0.23f, 0.23f, 1.0f), "■");
    ImGui::SameLine(); ImGui::Text("Traceback path");
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.2f, 0.7f, 0.2f, 1.0f), "■");
    ImGui::SameLine(); ImGui::Text("High score");
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.39f, 0.63f, 1.0f, 1.0f), "■");
    ImGui::SameLine(); ImGui::Text("Computing");
    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.24f, 0.24f, 0.27f, 1.0f), "■");
    ImGui::SameLine(); ImGui::Text("Pending");
}

