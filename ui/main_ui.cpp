#include "app_state.h"
#include "pattern_search.h"

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
    
    // Disable debug features that add "?" prefixes
    io.ConfigDebugIsDebuggerPresent = false;
    io.ConfigDebugHighlightIdConflicts = false;

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
                    static PatternSearchState ps_state;
                    ImGui::BeginChild("##ps_scroll", ImVec2(0, 0), false, 0);
                    render_pattern_search(ps_state);
                    ImGui::EndChild();
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
