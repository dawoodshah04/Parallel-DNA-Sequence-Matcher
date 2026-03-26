#include "app_state.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

#include "imgui.h"

// ── Color mapping for DP cell values ──────────────────────────────────────

static ImU32 cell_color(int val, int max_val, bool on_traceback) {
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
    for (std::size_t k = 0; k < len; ++k)
        mid[k] = (aq[k] != '-' && aq[k] == ar[k]) ? '|' : ' ';

    ImGui::Text("Query : %s", aq.c_str());
    ImGui::Text("       %s", mid.c_str());
    ImGui::Text("Ref   : %s", ar.c_str());
    ImGui::Spacing();
    ImGui::Text("Score : %d   |   Query start: %d   |   Ref start: %d",
                res.score, res.query_start, res.ref_start);
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

    ImGui::SliderFloat("Cell size (px)", &state.cell_size, 10.0f, 40.0f);
    ImGui::SliderInt("Max visible cells", &state.max_vis, 10, 60);
    ImGui::Spacing();

    bool run_clicked = ImGui::Button("Run Smith-Waterman", ImVec2(220, 0));
    if (run_clicked && std::strlen(state.query_buf) > 0 &&
                       std::strlen(state.ref_buf)   > 0) {
        state.viz = smith_waterman_full(state.query_buf, state.ref_buf);
        state.computed = true;
    }

    if (!state.computed) {
        ImGui::TextDisabled("Enter sequences above and press Run.");
        return;
    }

    // ── Alignment result ───────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::SeparatorText("Alignment Result");
    show_alignment(state.viz.result);

    // ── DP matrix heatmap ─────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::SeparatorText("DP Matrix (Smith-Waterman)");

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
    std::vector<std::vector<bool>> on_tb(
        static_cast<std::size_t>(m_vis + 1),
        std::vector<bool>(static_cast<std::size_t>(n_vis + 1), false));
    for (auto& [pi, pj] : state.viz.traceback_cells) {
        if (pi <= m_vis && pj <= n_vis)
            on_tb[static_cast<std::size_t>(pi)][static_cast<std::size_t>(pj)] = true;
    }

    ImVec2 canvas = ImGui::GetCursorScreenPos();
    float total_w = (n_vis + 1) * cell_sz;
    float total_h = (m_vis + 1) * cell_sz;

    // Reserve space so ImGui scrollbar appears when needed
    ImGui::InvisibleButton("##matrix_area", ImVec2(total_w + 4, total_h + 4));

    ImDrawList* dl = ImGui::GetWindowDrawList();

    for (int i = 0; i <= m_vis; ++i) {
        for (int j = 0; j <= n_vis; ++j) {
            float x0 = canvas.x + j * cell_sz;
            float y0 = canvas.y + i * cell_sz;
            float x1 = x0 + cell_sz - 1;
            float y1 = y0 + cell_sz - 1;

            bool tb = on_tb[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            ImU32 col = cell_color(H[i][j], max_val, tb);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), col);

            // Show value text when cells are large enough
            if (cell_sz >= 18) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "%d", H[i][j]);
                dl->AddText(ImVec2(x0 + 2, y0 + 2), IM_COL32_WHITE, buf);
            }
        }
    }

    // Row and column labels outside the grid
    for (int j = 1; j <= n_vis; ++j) {
        float x = canvas.x + j * cell_sz + 2;
        float y = canvas.y - cell_sz * 0.8f;
        char buf[4];
        std::snprintf(buf, sizeof(buf), "%c",
            state.ref_buf[static_cast<std::size_t>(j - 1)]);
        dl->AddText(ImVec2(x, y), IM_COL32(180, 180, 180, 255), buf);
    }
    for (int i = 1; i <= m_vis; ++i) {
        float x = canvas.x - cell_sz * 0.8f;
        float y = canvas.y + i * cell_sz + 2;
        char buf[4];
        std::snprintf(buf, sizeof(buf), "%c",
            state.query_buf[static_cast<std::size_t>(i - 1)]);
        dl->AddText(ImVec2(x, y), IM_COL32(180, 180, 180, 255), buf);
    }

    if (m_vis < m_full || n_vis < n_full) {
        ImGui::TextDisabled(
            "(showing %dx%d of %dx%d — increase 'Max visible cells' to see more)",
            m_vis, n_vis, m_full, n_full);
    }

    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.86f, 0.23f, 0.23f, 1.0f), "■");
    ImGui::SameLine(); ImGui::Text("= traceback path");
    ImGui::SameLine(); ImGui::Spacing();
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.2f, 0.7f, 0.2f, 1.0f), "■");
    ImGui::SameLine(); ImGui::Text("= high score");
}
