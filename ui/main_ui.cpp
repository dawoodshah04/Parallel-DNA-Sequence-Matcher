#include "app_state.h"

// GLFW + OpenGL3 ImGui backends
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>

// ── Error callback ─────────────────────────────────────────────────────────

static void glfw_error_callback(int error, const char* description) {
    std::fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// ── Entry point ────────────────────────────────────────────────────────────

int main(int /*argc*/, char** /*argv*/) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

    // OpenGL 3.3 core profile (works on WSLg + Intel iGPU and NVIDIA)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(
        1280, 800, "Parallel DNA Sequence Matcher", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // vsync

    // ── ImGui setup ────────────────────────────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();

    // Tweak style for readability
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding  = 6.0f;
    style.FrameRounding   = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.GrabRounding    = 4.0f;
    style.WindowPadding   = ImVec2(12, 10);
    style.FramePadding    = ImVec2(6, 4);
    style.ItemSpacing     = ImVec2(8, 6);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ── Application state (persists across frames) ─────────────────────────
    SWVisualizerState sw_state;
    BenchmarkState    bm_state;

    // ── Main loop ──────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Full-screen dockspace / background window
        {
            int fb_w, fb_h;
            glfwGetFramebufferSize(window, &fb_w, &fb_h);
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(static_cast<float>(fb_w),
                                            static_cast<float>(fb_h)));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::Begin("##root", nullptr,
                ImGuiWindowFlags_NoTitleBar
                | ImGuiWindowFlags_NoResize
                | ImGuiWindowFlags_NoScrollbar
                | ImGuiWindowFlags_NoBringToFrontOnFocus
                | ImGuiWindowFlags_MenuBar);
            ImGui::PopStyleVar();

            // ── Menu bar ──────────────────────────────────────────────────
            if (ImGui::BeginMenuBar()) {
                ImGui::Text("Parallel DNA Sequence Matcher");
                ImGui::SameLine(ImGui::GetContentRegionAvail().x - 120.0f);
                ImGui::TextDisabled("OpenMP | MPI | OpenCL");
                ImGui::EndMenuBar();
            }

            // ── Tab bar ───────────────────────────────────────────────────
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4);
            if (ImGui::BeginTabBar("##tabs")) {

                // ── Tab 1: Smith-Waterman Visualizer ──────────────────────
                if (ImGui::BeginTabItem("SW Alignment Visualizer")) {
                    ImGui::BeginChild("##sw_scroll", ImVec2(0, 0), false,
                                     ImGuiWindowFlags_HorizontalScrollbar);
                    render_sw_visualizer(sw_state);
                    ImGui::EndChild();
                    ImGui::EndTabItem();
                }

                // ── Tab 2: Benchmark Panel ────────────────────────────────
                if (ImGui::BeginTabItem("Benchmark")) {
                    ImGui::BeginChild("##bm_scroll", ImVec2(0, 0), false, 0);
                    render_benchmark_panel(bm_state);
                    ImGui::EndChild();
                    ImGui::EndTabItem();
                }

                // ── Tab 3: Pattern Search ─────────────────────────────────
                if (ImGui::BeginTabItem("Pattern Search")) {
                    static char txt_buf[512] = "ACGTACGTACGTACGTACGTACGT";
                    static char pat_buf[128] = "ACGT";
                    static bool use_bm       = false;
                    static std::vector<int> hits;
                    static bool searched     = false;

                    ImGui::SeparatorText("Input");
                    ImGui::SetNextItemWidth(-1.0f);
                    ImGui::InputText("Text (reference)##ps", txt_buf, sizeof(txt_buf));
                    ImGui::SetNextItemWidth(300.0f);
                    ImGui::InputText("Pattern##ps", pat_buf, sizeof(pat_buf));
                    ImGui::SameLine();
                    ImGui::Checkbox("Boyer-Moore", &use_bm);
                    ImGui::SameLine();
                    if (ImGui::Button("Search##ps")) {
                        if (use_bm) hits = boyer_moore_search(txt_buf, pat_buf);
                        else        hits = kmp_search(txt_buf, pat_buf);
                        searched = true;
                    }

                    if (searched) {
                        ImGui::Spacing();
                        ImGui::SeparatorText("Results");
                        ImGui::Text("Algorithm : %s", use_bm ? "Boyer-Moore" : "KMP");
                        ImGui::Text("Hits      : %zu", hits.size());
                        if (!hits.empty()) {
                            ImGui::Text("Positions : ");
                            for (std::size_t k = 0; k < hits.size() && k < 50; ++k) {
                                ImGui::SameLine();
                                ImGui::Text("%d", hits[k]);
                                if (k + 1 < hits.size() && k + 1 < 50)
                                    ImGui::SameLine();
                                ImGui::Text(",");
                            }
                            if (hits.size() > 50) {
                                ImGui::SameLine();
                                ImGui::TextDisabled("... and %zu more", hits.size() - 50);
                            }
                        }
                    }
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            ImGui::End();
        }

        // ── Render ─────────────────────────────────────────────────────────
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.10f, 0.10f, 0.12f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // ── Cleanup ────────────────────────────────────────────────────────────
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
