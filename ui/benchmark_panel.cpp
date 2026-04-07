#include "app_state.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "imgui.h"
#include <omp.h>

// Define OpenCL target version to suppress warning
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>

#ifndef _WIN32
#  include <unistd.h>   // readlink
#else
#  include <windows.h>
#endif

// ── Exe-directory helper ───────────────────────────────────────────────────────

static std::string get_exe_dir() {
#ifndef _WIN32
    char buf[4096] = {};
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) buf[n] = '\0';
    return std::filesystem::path(buf).parent_path().string();
#else
    char buf[4096] = {};
    GetModuleFileNameA(nullptr, buf, sizeof(buf));
    return std::filesystem::path(buf).parent_path().string();
#endif
}

// ── In-process benchmark helpers ──────────────────────────────────────────

// Run SW over `db_size` random sequences vs one query, returning elapsed ms.
static double run_sw_sequential(int seq_len, int db_size, int& out_score) {
    std::string query = generate_random_sequence(seq_len, /*seed=*/42);
    std::vector<std::string> db(static_cast<std::size_t>(db_size));
    for (int i = 0; i < db_size; ++i)
        db[static_cast<std::size_t>(i)] = generate_random_sequence(seq_len, 100u + static_cast<unsigned>(i));

    BenchmarkTimer timer;
    timer.start();
    int best = 0;
    for (int i = 0; i < db_size; ++i) {
        auto r = smith_waterman(query, db[static_cast<std::size_t>(i)]);
        if (r.score > best) best = r.score;
    }
    timer.stop();
    out_score = best;
    return timer.elapsed_ms();
}

static double run_sw_openmp(int seq_len, int db_size, int threads, int& out_score) {
    std::string query = generate_random_sequence(seq_len, 42u);
    std::vector<std::string> db(static_cast<std::size_t>(db_size));
    for (int i = 0; i < db_size; ++i)
        db[static_cast<std::size_t>(i)] = generate_random_sequence(seq_len, 100u + static_cast<unsigned>(i));

    std::vector<int> scores(static_cast<std::size_t>(db_size), 0);

    omp_set_num_threads(threads);
    double t_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < db_size; ++i) {
        auto r = smith_waterman(query, db[static_cast<std::size_t>(i)]);
        scores[static_cast<std::size_t>(i)] = r.score;
    }

    double t_end = omp_get_wtime();

    int best = *std::max_element(scores.begin(), scores.end());
    out_score = best;
    return (t_end - t_start) * 1000.0;
}
// ── OpenCL in-process SW benchmark ─────────────────────────────────────────────

static double run_sw_opencl(int seq_len, int db_size, int& out_score,
                             std::string& err_out) {
    // Discover platform / device
    cl_uint num_plat = 0;
    if (clGetPlatformIDs(0, nullptr, &num_plat) != CL_SUCCESS || num_plat == 0) {
        err_out = "No OpenCL platform.\nFix: sudo apt install pocl-opencl-icd";
        return -1.0;
    }
    std::vector<cl_platform_id> platforms(num_plat);
    clGetPlatformIDs(num_plat, platforms.data(), nullptr);

    cl_device_id device = nullptr;
    for (auto p : platforms)
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) == CL_SUCCESS) break;
    if (!device)
        for (auto p : platforms)
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 1, &device, nullptr) == CL_SUCCESS) break;
    if (!device) { err_out = "No OpenCL device available."; return -1.0; }

    // Load sw_kernel.cl from same directory as the running executable
    std::string kernel_path =
        (std::filesystem::path(get_exe_dir()) / "sw_kernel.cl").string();
    std::string src;
    {
        std::ifstream f(kernel_path);
        if (!f.is_open()) {
            err_out = "sw_kernel.cl not found:\n" + kernel_path;
            return -1.0;
        }
        std::ostringstream ss; ss << f.rdbuf(); src = ss.str();
    }

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) { err_out = "clCreateContext failed"; return -1.0; }

    cl_queue_properties queue_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue =
        clCreateCommandQueueWithProperties(ctx, device, queue_props, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(ctx);
        err_out = "clCreateCommandQueue failed";
        return -1.0;
    }

    const char* src_ptr = src.c_str();
    std::size_t src_len = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    if (clBuildProgram(prog, 1, &device, "-cl-std=CL1.2", nullptr, nullptr)
            != CL_SUCCESS) {
        char log[4096] = {};
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, nullptr);
        err_out = std::string("Kernel build error:\n") + log;
        clReleaseProgram(prog); clReleaseCommandQueue(queue); clReleaseContext(ctx);
        return -1.0;
    }
    
    // Check device capabilities to choose optimized or fallback kernel
    cl_ulong max_work_group_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                    sizeof(max_work_group_size), &max_work_group_size, nullptr);
    
    int m = seq_len, n = seq_len;
    int max_diag_len = (m < n) ? m : n;
    bool use_optimized = (max_diag_len <= (int)max_work_group_size);
    
    const char* kernel_name = use_optimized ? "sw_wavefront_optimized" : "sw_wavefront";
    cl_kernel sw_kern = clCreateKernel(prog, kernel_name, &err);
    if (err != CL_SUCCESS) {
        // Fallback to old kernel if optimized not available
        sw_kern = clCreateKernel(prog, "sw_wavefront", &err);
        use_optimized = false;
    }
    cl_kernel fm_kern = clCreateKernel(prog, "find_max", &err);

    // Use identical seeds as sequential/openmp runs for fair comparison
    std::string query = generate_random_sequence(seq_len, 42u);
    std::vector<std::string> db(static_cast<std::size_t>(db_size));
    for (int i = 0; i < db_size; ++i)
        db[static_cast<std::size_t>(i)] =
            generate_random_sequence(seq_len, 100u + static_cast<unsigned>(i));

    cl_int sw_m = m, sw_n = n;
    cl_int sw_match = 2, sw_mm = -1, sw_gap = -2;
    std::size_t h_elems = static_cast<std::size_t>((m + 1) * (n + 1));

    cl_mem q_buf = clCreateBuffer(ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        query.size(), const_cast<char*>(query.data()), &err);

    double total_ms = 0.0;
    int    best_score = 0;

    for (int i = 0; i < db_size; ++i) {
        std::vector<cl_int> H_host(h_elems, 0);
        cl_mem H_buf = clCreateBuffer(ctx,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            h_elems * sizeof(cl_int), H_host.data(), &err);
        cl_mem r_buf = clCreateBuffer(ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            db[static_cast<std::size_t>(i)].size(),
            const_cast<char*>(db[static_cast<std::size_t>(i)].data()), &err);
        cl_mem score_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            sizeof(cl_int), nullptr, &err);
        cl_mem row_buf   = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            sizeof(cl_int), nullptr, &err);
        cl_mem col_buf   = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            sizeof(cl_int), nullptr, &err);

        std::vector<cl_event> wave_events;
        cl_event first_ev, last_ev;
        
        if (use_optimized) {
            // Single-kernel optimized path
            clSetKernelArg(sw_kern, 0, sizeof(H_buf),  &H_buf);
            clSetKernelArg(sw_kern, 1, sizeof(q_buf),  &q_buf);
            clSetKernelArg(sw_kern, 2, sizeof(r_buf),  &r_buf);
            clSetKernelArg(sw_kern, 3, sizeof(cl_int), &sw_m);
            clSetKernelArg(sw_kern, 4, sizeof(cl_int), &sw_n);
            clSetKernelArg(sw_kern, 5, sizeof(cl_int), &sw_match);
            clSetKernelArg(sw_kern, 6, sizeof(cl_int), &sw_mm);
            clSetKernelArg(sw_kern, 7, sizeof(cl_int), &sw_gap);
            
            std::size_t global_size = static_cast<std::size_t>(max_diag_len);
            clEnqueueNDRangeKernel(queue, sw_kern, 1,
                                   nullptr, &global_size, nullptr, 0, nullptr, &first_ev);
            last_ev = first_ev;
        } else {
            // Multi-kernel fallback path
            clSetKernelArg(sw_kern, 0, sizeof(H_buf),  &H_buf);
            clSetKernelArg(sw_kern, 1, sizeof(q_buf),  &q_buf);
            clSetKernelArg(sw_kern, 2, sizeof(r_buf),  &r_buf);
            clSetKernelArg(sw_kern, 3, sizeof(cl_int), &sw_m);
            clSetKernelArg(sw_kern, 4, sizeof(cl_int), &sw_n);
            // arg 5 = diag (set per wavefront step below)
            clSetKernelArg(sw_kern, 6, sizeof(cl_int), &sw_match);
            clSetKernelArg(sw_kern, 7, sizeof(cl_int), &sw_mm);
            clSetKernelArg(sw_kern, 8, sizeof(cl_int), &sw_gap);

            wave_events.reserve(static_cast<std::size_t>(m + n));

            for (int diag = 2; diag <= m + n; ++diag) {
                int i_min = std::max(1, diag - n);
                int i_max = std::min(m, diag - 1);
                if (i_min > i_max) continue;
                std::size_t global = static_cast<std::size_t>(i_max - i_min + 1);
                cl_int diag_arg = diag;
                clSetKernelArg(sw_kern, 5, sizeof(cl_int), &diag_arg);
                cl_event ev;
                clEnqueueNDRangeKernel(queue, sw_kern, 1,
                                       nullptr, &global, nullptr, 0, nullptr, &ev);
                wave_events.push_back(ev);
            }
            if (!wave_events.empty()) {
                first_ev = wave_events.front();
                last_ev = wave_events.back();
            }
        }

        cl_int fm_rows = m + 1, fm_cols = n + 1;
        clSetKernelArg(fm_kern, 0, sizeof(H_buf),     &H_buf);
        clSetKernelArg(fm_kern, 1, sizeof(cl_int),    &fm_rows);
        clSetKernelArg(fm_kern, 2, sizeof(cl_int),    &fm_cols);
        clSetKernelArg(fm_kern, 3, sizeof(score_buf), &score_buf);
        clSetKernelArg(fm_kern, 4, sizeof(row_buf),   &row_buf);
        clSetKernelArg(fm_kern, 5, sizeof(col_buf),   &col_buf);

        std::size_t one = 1;
        cl_event fm_ev;
        clEnqueueNDRangeKernel(queue, fm_kern, 1,
                               nullptr, &one, nullptr, 0, nullptr, &fm_ev);
        clFinish(queue);

        cl_int sc = 0;
        clEnqueueReadBuffer(queue, score_buf, CL_TRUE, 0,
                            sizeof(cl_int), &sc, 0, nullptr, nullptr);
        if (sc > best_score) best_score = sc;

        // Accumulate GPU time: first kernel start → find_max end
        cl_ulong t0 = 0, t1 = 0;
        if (use_optimized || !wave_events.empty()) {
            clGetEventProfilingInfo(first_ev,
                CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        }
        clGetEventProfilingInfo(fm_ev,
            CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        total_ms += static_cast<double>(t1 - t0) / 1.0e6;

        // Cleanup events
        if (use_optimized) {
            clReleaseEvent(first_ev);
        } else {
            for (auto& ev : wave_events) clReleaseEvent(ev);
        }
        clReleaseEvent(fm_ev);
        clReleaseMemObject(H_buf);
        clReleaseMemObject(r_buf);
        clReleaseMemObject(score_buf);
        clReleaseMemObject(row_buf);
        clReleaseMemObject(col_buf);
    }

    out_score = best_score;
    clReleaseMemObject(q_buf);
    clReleaseKernel(sw_kern);
    clReleaseKernel(fm_kern);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return total_ms;
}

// ── MPI subprocess SW benchmark ───────────────────────────────────────────────

static double run_sw_mpi(int seq_len, int db_size, int ranks,
                          int& out_score, std::string& err_out) {
    // Write a temp FASTA with the same seeds used by other variants
    const std::string tmp_path = "/tmp/dna_ui_bench.fasta";
    {
        std::ofstream f(tmp_path);
        if (!f.is_open()) {
            err_out = "Cannot write " + tmp_path;
            return -1.0;
        }
        f << ">query_bench\n" << generate_random_sequence(seq_len, 42u) << "\n";
        for (int i = 0; i < db_size; ++i)
            f << ">db_seq_" << i << "\n"
              << generate_random_sequence(seq_len, 100u + static_cast<unsigned>(i))
              << "\n";
    }

    std::string mpi_exe =
        (std::filesystem::path(get_exe_dir()) / "dna_mpi").string();

    char cmd[2048];
    std::snprintf(cmd, sizeof(cmd), "mpiexec --oversubscribe -n %d \"%s\" \"%s\" 2>&1",
                  ranks, mpi_exe.c_str(), tmp_path.c_str());

#ifdef _WIN32
    FILE* pipe = _popen(cmd, "r");
#else
    FILE* pipe = popen(cmd, "r");
#endif
    if (!pipe) {
        err_out = "popen failed — is mpiexec in $PATH?";
        std::remove(tmp_path.c_str());
        return -1.0;
    }

    char   line[512];
    double mpi_ms    = -1.0;
    int    mpi_score = 0;
    while (std::fgets(line, sizeof(line), pipe)) {
        double val; int sc;
        if (std::sscanf(line, "[MPI] SW Time: %lf ms", &val) == 1)
            mpi_ms = val;
        if (std::sscanf(line, "Global best score: %d", &sc) == 1)
            mpi_score = sc;
    }

#ifdef _WIN32
    int exit_code = _pclose(pipe);
#else
    int exit_code = pclose(pipe);
#endif
    std::remove(tmp_path.c_str());

    if (mpi_ms < 0.0) {
        if (exit_code != 0)
            err_out = "mpiexec exited " + std::to_string(exit_code) +
                      " — ensure dna_mpi is built.";
        else
            err_out = "Could not parse [MPI] SW Time from output.";
        return -1.0;
    }

    out_score = mpi_score;
    return mpi_ms;
}
// ── Bar chart drawing helper ───────────────────────────────────────────────

static void draw_bar_chart(const std::vector<BenchmarkEntry>& entries,
                            double seq_base_ms) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 cursor  = ImGui::GetCursorScreenPos();
    float avail_w  = ImGui::GetContentRegionAvail().x - 160.0f;  // label space

    // Find max timing for scaling
    double max_t = 1.0;
    for (auto& e : entries)
        if (e.measured && e.time_ms > max_t) max_t = e.time_ms;

    constexpr float BAR_H    = 28.0f;  // Increased height
    constexpr float BAR_PAD  = 10.0f;  // Increased padding
    constexpr float LABEL_W  = 120.0f;

    // Modern gradient colors
    ImVec4 colors[] = {
        {0.40f, 0.70f, 1.00f, 1.0f},   // sequential — bright blue
        {0.30f, 0.85f, 0.55f, 1.0f},   // omp 2t — bright green
        {0.25f, 0.75f, 0.45f, 1.0f},   // omp 4t
        {0.20f, 0.65f, 0.35f, 1.0f},   // omp 8t
        {1.00f, 0.65f, 0.20f, 1.0f},   // MPI — bright orange
        {0.80f, 0.40f, 1.00f, 1.0f},   // OpenCL — bright purple
    };

    // Icons for each variant
    const char* icons[] = {"⚡", "⚙", "⚙", "⚙", "📡", "🖥"};

    for (std::size_t idx = 0; idx < entries.size(); ++idx) {
        const auto& e = entries[idx];
        float y = cursor.y + static_cast<float>(idx) * (BAR_H + BAR_PAD);

        // Icon + Label with better formatting
        ImGui::SetCursorScreenPos(ImVec2(cursor.x, y + 4));
        ImGui::PushStyleColor(ImGuiCol_Text, colors[idx % 6]);
        ImGui::Text("%s", icons[idx % 6]);
        ImGui::PopStyleColor();
        
        ImGui::SameLine();
        ImGui::Text("%-12s", e.label);

        if (!e.measured) {
            ImGui::SetCursorScreenPos(ImVec2(cursor.x + LABEL_W, y + 4));
            if (!e.error.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
                ImGui::Text("❌ %s", e.error.c_str());
                ImGui::PopStyleColor();
            } else {
                ImGui::TextDisabled("(not run)");
            }
            continue;
        }

        float bar_w = static_cast<float>(e.time_ms / max_t) * avail_w;
        ImVec4 cv   = colors[idx % 6];
        
        // Create gradient effect
        ImVec4 cv_dark = ImVec4(cv.x * 0.7f, cv.y * 0.7f, cv.z * 0.7f, cv.w);
        ImU32 col_light = ImGui::ColorConvertFloat4ToU32(cv);
        ImU32 col_dark = ImGui::ColorConvertFloat4ToU32(cv_dark);

        float bx0 = cursor.x + LABEL_W;
        float bx1 = bx0 + std::max(bar_w, 4.0f);

        // Draw gradient bar with rounded corners
        dl->AddRectFilledMultiColor(
            ImVec2(bx0, y + 2), ImVec2(bx1, y + BAR_H),
            col_light, col_dark, col_dark, col_light
        );
        
        // Add subtle border
        dl->AddRect(ImVec2(bx0, y + 2), ImVec2(bx1, y + BAR_H),
                   IM_COL32(255, 255, 255, 60), 4.0f, 0, 1.5f);

        // Speedup annotation with better formatting
        char info[64];
        if (seq_base_ms > 0.0 && e.time_ms > 0.0) {
            double speedup = seq_base_ms / e.time_ms;
            if (speedup > 1.5) {
                std::snprintf(info, sizeof(info), "%.1f ms  (🚀 %.2fx faster)",
                              e.time_ms, speedup);
            } else if (speedup < 0.8) {
                std::snprintf(info, sizeof(info), "%.1f ms  (🐢 %.2fx slower)",
                              e.time_ms, 1.0/speedup);
            } else {
                std::snprintf(info, sizeof(info), "%.1f ms  (≈ %.2fx)",
                              e.time_ms, speedup);
            }
        } else {
            std::snprintf(info, sizeof(info), "%.1f ms", e.time_ms);
        }
        
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        dl->AddText(ImVec2(bx1 + 8, y + 6), IM_COL32_WHITE, info);
        ImGui::PopStyleColor();
    }

    // Advance cursor past the bars
    ImGui::Dummy(ImVec2(avail_w + LABEL_W,
                        static_cast<float>(entries.size()) * (BAR_H + BAR_PAD) + 10.0f));
}

// ── Main render function ───────────────────────────────────────────────────

void render_benchmark_panel(BenchmarkState& state) {
    // ── Configuration Section ──────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
    ImGui::SeparatorText("⚙ Configuration");
    ImGui::PopStyleColor();

    ImGui::Columns(3, "config_cols", false);
    
    ImGui::Text("📏 Sequence Length");
    ImGui::SetNextItemWidth(-1);
    ImGui::SliderInt("##seq_len", &state.seq_length, 50, 1000, "%d bp");
    
    ImGui::NextColumn();
    ImGui::Text("🗄 Database Size");
    ImGui::SetNextItemWidth(-1);
    ImGui::SliderInt("##db_size", &state.db_size, 5, 200, "%d sequences");
    
    ImGui::NextColumn();
    ImGui::Text("📡 MPI Ranks");
    ImGui::SetNextItemWidth(-1);
    ImGui::SliderInt("##mpi_ranks", &state.mpi_ranks, 1, 8, "%d ranks");
    
    ImGui::Columns(1);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ── Run Button ─────────────────────────────────────────────────────────
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.4f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.15f, 0.6f, 0.25f, 1.0f));
    bool run = ImGui::Button("🚀 Run Benchmark", ImVec2(180, 35));
    ImGui::PopStyleColor(3);
    
    ImGui::SameLine();
    ImGui::TextDisabled("Sequential + OpenMP (in-process)  |  MPI (subprocess)  |  OpenCL (in-process)");

    if (run && !state.running) {
        state.running = true;

        // Sequential
        {
            int sc = 0;
            double ms = run_sw_sequential(state.seq_length, state.db_size, sc);
            state.entries[0] = {"Sequential", ms, sc, true, ""};
        }
        // OpenMP — 2 threads
        {
            int sc = 0;
            double ms = run_sw_openmp(state.seq_length, state.db_size, 2, sc);
            state.entries[1] = {"OpenMP 2T", ms, sc, true, ""};
        }
        // OpenMP — 4 threads
        {
            int sc = 0;
            double ms = run_sw_openmp(state.seq_length, state.db_size, 4, sc);
            state.entries[2] = {"OpenMP 4T", ms, sc, true, ""};
        }
        // OpenMP — 8 threads
        {
            int sc = 0;
            double ms = run_sw_openmp(state.seq_length, state.db_size, 8, sc);
            state.entries[3] = {"OpenMP 8T", ms, sc, true, ""};
        }
        // MPI — subprocess via mpiexec
        {
            int sc = 0; std::string emsg;
            double ms = run_sw_mpi(state.seq_length, state.db_size,
                                   state.mpi_ranks, sc, emsg);
            if (ms >= 0.0) state.entries[4] = {"MPI",    ms,  sc, true,  ""};
            else           state.entries[4] = {"MPI",    0.0, 0,  false, emsg};
        }
        // OpenCL — in-process via OpenCL C API
        {
            int sc = 0; std::string emsg;
            double ms = run_sw_opencl(state.seq_length, state.db_size, sc, emsg);
            if (ms >= 0.0) state.entries[5] = {"OpenCL", ms,  sc, true,  ""};
            else           state.entries[5] = {"OpenCL", 0.0, 0,  false, emsg};
        }

        state.running = false;
    }

    // ── Results Section ────────────────────────────────────────────────────
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.3f, 1.0f));
    ImGui::SeparatorText("📊 Performance Results");
    ImGui::PopStyleColor();

    double base_ms = state.entries[0].measured ? state.entries[0].time_ms : 0.0;
    
    // Show summary stats if benchmarks have run
    if (state.entries[0].measured) {
        int successful = 0;
        double fastest = 999999.0;
        const char* fastest_name = "";
        
        for (const auto& e : state.entries) {
            if (e.measured) {
                successful++;
                if (e.time_ms < fastest) {
                    fastest = e.time_ms;
                    fastest_name = e.label;
                }
            }
        }
        
        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.2f, 0.25f, 0.5f));
        ImGui::BeginChild("##stats", ImVec2(-1, 60), true);
        
        ImGui::Columns(3, "summary", false);
        ImGui::Text("✅ Successful: %d / %zu", successful, state.entries.size());
        
        ImGui::NextColumn();
        ImGui::Text("⏱ Baseline: %.2f ms", base_ms);
        
        ImGui::NextColumn();
        ImGui::Text("🏆 Fastest: %s (%.2f ms)", fastest_name, fastest);
        
        ImGui::Columns(1);
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::EndGroup();
        ImGui::Spacing();
    }
    
    draw_bar_chart(state.entries, base_ms);

    // ── Speedup Table ──────────────────────────────────────────────────────
    if (state.entries[0].measured) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 1.0f, 0.7f, 1.0f));
        ImGui::SeparatorText("📈 Detailed Speedup Analysis");
        ImGui::PopStyleColor();
        
        if (ImGui::BeginTable("speedup_tbl", 4,
                ImGuiTableFlags_Borders | 
                ImGuiTableFlags_RowBg | 
                ImGuiTableFlags_Resizable)) {
            
            ImGui::TableSetupColumn("Implementation", ImGuiTableColumnFlags_WidthFixed, 120);
            ImGui::TableSetupColumn("Time (ms)", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("Speedup", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("Efficiency", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            for (auto& e : state.entries) {
                ImGui::TableNextRow();
                
                // Variant name
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%s", e.label);
                
                // Time
                ImGui::TableSetColumnIndex(1);
                if (e.measured)
                    ImGui::Text("%.3f", e.time_ms);
                else if (!e.error.empty())
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "error");
                else
                    ImGui::TextDisabled("—");
                
                // Speedup
                ImGui::TableSetColumnIndex(2);
                if (e.measured && base_ms > 0.0) {
                    double speedup = base_ms / e.time_ms;
                    if (speedup >= 2.0)
                        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%.2fx", speedup);
                    else if (speedup >= 1.2)
                        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.3f, 1.0f), "%.2fx", speedup);
                    else
                        ImGui::Text("%.2fx", speedup);
                } else {
                    ImGui::TextDisabled("—");
                }
                
                // Efficiency bar
                ImGui::TableSetColumnIndex(3);
                if (e.measured && base_ms > 0.0) {
                    double speedup = base_ms / e.time_ms;
                    double efficiency = speedup / (e.label[0] == 'S' ? 1.0 : 
                                       e.label[0] == 'O' ? (e.label[7] - '0') : 
                                       4.0); // Default for MPI/OpenCL
                    
                    float bar_width = std::min(static_cast<float>(efficiency), 1.0f);
                    ImVec4 bar_color = efficiency > 0.8f ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) :
                                      efficiency > 0.5f ? ImVec4(0.8f, 0.8f, 0.2f, 1.0f) :
                                                         ImVec4(0.8f, 0.3f, 0.2f, 1.0f);
                    
                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_color);
                    ImGui::ProgressBar(bar_width, ImVec2(-1, 0), "");
                    ImGui::PopStyleColor();
                } else {
                    ImGui::TextDisabled("—");
                }
            }
            ImGui::EndTable();
        }
    }
}
