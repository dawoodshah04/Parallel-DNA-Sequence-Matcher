#include <CL/cl.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "dna_utils.h"
#include "smith_waterman.h"  // SW_MATCH, SW_MISMATCH, SW_GAP
#include "kmp.h"

// ── CL_CHECK macro ─────────────────────────────────────────────────────────
#define CL_CHECK(err)                                                        \
    do {                                                                     \
        if ((err) != CL_SUCCESS) {                                           \
            std::fprintf(stderr, "OpenCL error %d at %s:%d\n",              \
                         static_cast<int>(err), __FILE__, __LINE__);        \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// ── Helpers ────────────────────────────────────────────────────────────────

static std::string load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static cl_device_id pick_device(std::vector<cl_platform_id>& platforms) {
    // Prefer GPU; fall back to CPU
    cl_device_id dev = nullptr;
    for (auto plat : platforms) {
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) == CL_SUCCESS)
            return dev;
    }
    for (auto plat : platforms) {
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 1, &dev, nullptr) == CL_SUCCESS)
            return dev;
    }
    return nullptr;
}

static cl_program build_program(cl_context ctx, cl_device_id dev,
                                 const std::string& src,
                                 const char* options = "-cl-std=CL1.2") {
    cl_int err;
    const char* src_ptr = src.c_str();
    std::size_t src_len = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    CL_CHECK(err);

    if (clBuildProgram(prog, 1, &dev, options, nullptr, nullptr) != CL_SUCCESS) {
        char log[8192] = {};
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, nullptr);
        std::cerr << "Build error:\n" << log << '\n';
        std::exit(1);
    }
    return prog;
}

// ── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: dna_opencl <fasta_file> [pattern_length]\n";
        std::cerr << "  pattern_length: Length of pattern to search (default: min(query_len, 50))\n";
        return 1;
    }
    
    // Parse optional pattern length argument
    int pattern_length = -1;  // -1 means auto-detect
    if (argc >= 3) {
        try {
            pattern_length = std::stoi(argv[2]);
            if (pattern_length <= 0) {
                std::cerr << "Error: pattern_length must be positive\n";
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing pattern_length: " << e.what() << '\n';
            return 1;
        }
    }

    // ── Load sequences ─────────────────────────────────────────────────────
    std::vector<DNASequence> sequences;
    try {
        sequences = parse_fasta(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    if (sequences.size() < 2) {
        std::cerr << "Error: FASTA file must have at least 2 sequences.\n";
        return 1;
    }

    const std::string& query = sequences[0].sequence;
    const std::string& ref   = sequences[1].sequence;
    int m = static_cast<int>(query.size());
    int n = static_cast<int>(ref.size());

    std::cout << "Query : " << sequences[0].id << " (" << m << " bp)\n";
    std::cout << "Ref   : " << sequences[1].id << " (" << n << " bp)\n\n";

    // ── Locate kernel files (same directory as executable) ─────────────────
    std::filesystem::path exe_dir =
        std::filesystem::path(argv[0]).parent_path();
    std::string sw_src  = load_file((exe_dir / "sw_kernel.cl").string());
    std::string kmp_src = load_file((exe_dir / "kmp_kernel.cl").string());

    // ── OpenCL platform + device ───────────────────────────────────────────
    cl_uint num_plat = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_plat));
    if (num_plat == 0) { std::cerr << "No OpenCL platforms found.\n"; return 1; }

    std::vector<cl_platform_id> platforms(num_plat);
    CL_CHECK(clGetPlatformIDs(num_plat, platforms.data(), nullptr));

    cl_device_id device = pick_device(platforms);
    if (!device) { std::cerr << "No OpenCL device found.\n"; return 1; }

    char dev_name[256] = {};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    std::cout << "OpenCL device : " << dev_name << "\n";
    
    // Query device memory capacity
    cl_ulong device_mem_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 
                    sizeof(device_mem_size), &device_mem_size, nullptr);
    std::cout << "Device memory : " << (device_mem_size / (1024*1024)) << " MB\n\n";
    
    // Calculate required memory for Smith-Waterman
    std::size_t h_elems = static_cast<std::size_t>((m + 1) * (n + 1));
    std::size_t sw_mem_required = h_elems * sizeof(cl_int) +  // H matrix
                                   m + n +                      // query + ref sequences
                                   3 * sizeof(cl_int);          // score, row, col
    
    // Calculate required memory for pattern search
    std::size_t kmp_mem_required = n +                         // text
                                    50 +                        // pattern (max)
                                    n * sizeof(cl_int) +        // matches array
                                    sizeof(cl_int);             // total
    
    std::size_t total_mem_required = sw_mem_required + kmp_mem_required;
    
    // Check if we have enough device memory (leave 10% margin for driver overhead)
    if (total_mem_required > static_cast<std::size_t>(device_mem_size * 0.9)) {
        std::cerr << "Error: Sequences too large for device memory!\n";
        std::cerr << "  Required: " << (total_mem_required / (1024*1024)) << " MB\n";
        std::cerr << "  Available: " << (device_mem_size / (1024*1024)) << " MB\n";
        std::cerr << "  Consider using smaller sequences or upgrading GPU.\n";
        return 1;
    }
    
    std::cout << "[INFO] Memory usage: " << (total_mem_required / (1024*1024)) 
              << " MB / " << (device_mem_size / (1024*1024)) << " MB\n\n";

    // ── Context + command queue (profiling enabled) ────────────────────────
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);

    cl_queue_properties queue_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue =
        clCreateCommandQueueWithProperties(ctx, device, queue_props, &err);
    CL_CHECK(err);

    // ══════════════════════════════════════════════════════════════════════
    // Smith-Waterman (wavefront)
    // ══════════════════════════════════════════════════════════════════════

    cl_program sw_prog = build_program(ctx, device, sw_src);
    
    // Query device capabilities for optimization strategy
    cl_ulong max_work_group_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                    sizeof(max_work_group_size), &max_work_group_size, nullptr);
    
    int max_diag_len = (m < n) ? m : n;
    bool use_optimized = (max_diag_len <= (int)max_work_group_size);
    
    // Select kernel based on sequence size
    const char* kernel_name = use_optimized ? "sw_wavefront_optimized" : "sw_wavefront";
    cl_kernel sw_kern = clCreateKernel(sw_prog, kernel_name, &err); 
    CL_CHECK(err);
    cl_kernel findmax_kern = clCreateKernel(sw_prog, "find_max", &err); 
    CL_CHECK(err);

    // Allocate DP matrix (zero-initialised) - reuse h_elems from memory check
    std::vector<cl_int> H_host(h_elems, 0);

    cl_mem H_buf  = clCreateBuffer(ctx,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        h_elems * sizeof(cl_int), H_host.data(), &err); CL_CHECK(err);
    cl_mem q_buf  = clCreateBuffer(ctx,
        CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
        query.size(), const_cast<char*>(query.data()), &err); CL_CHECK(err);
    cl_mem r_buf  = clCreateBuffer(ctx,
        CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
        ref.size(),   const_cast<char*>(ref.data()),   &err); CL_CHECK(err);

    cl_mem score_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &err); CL_CHECK(err);
    cl_mem row_buf   = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &err); CL_CHECK(err);
    cl_mem col_buf   = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &err); CL_CHECK(err);

    cl_int sw_m = m, sw_n = n;
    cl_int sw_match = SW_MATCH, sw_mm = SW_MISMATCH, sw_gap = SW_GAP;
    
    std::vector<cl_event> wave_events;
    cl_event sw_event;
    
    if (use_optimized) {
        // Single-kernel optimized path
        std::cout << "[INFO] Using optimized single-kernel SW (max_diag=" << max_diag_len << ")\n";
        
        CL_CHECK(clSetKernelArg(sw_kern, 0, sizeof(H_buf),  &H_buf));
        CL_CHECK(clSetKernelArg(sw_kern, 1, sizeof(q_buf),  &q_buf));
        CL_CHECK(clSetKernelArg(sw_kern, 2, sizeof(r_buf),  &r_buf));
        CL_CHECK(clSetKernelArg(sw_kern, 3, sizeof(cl_int), &sw_m));
        CL_CHECK(clSetKernelArg(sw_kern, 4, sizeof(cl_int), &sw_n));
        CL_CHECK(clSetKernelArg(sw_kern, 5, sizeof(cl_int), &sw_match));
        CL_CHECK(clSetKernelArg(sw_kern, 6, sizeof(cl_int), &sw_mm));
        CL_CHECK(clSetKernelArg(sw_kern, 7, sizeof(cl_int), &sw_gap));
        
        std::size_t global_size = static_cast<std::size_t>(max_diag_len);
        CL_CHECK(clEnqueueNDRangeKernel(queue, sw_kern, 1,
                                         nullptr, &global_size, nullptr, 0, nullptr, &sw_event));
        wave_events.push_back(sw_event);
    } else {
        // Multi-kernel fallback for large sequences
        std::cout << "[INFO] Using multi-kernel SW fallback (max_diag=" << max_diag_len << ")\n";
        
        // Static kernel args (indices 0-4, 6-8 — index 5 = diag, set per iteration)
        CL_CHECK(clSetKernelArg(sw_kern, 0, sizeof(H_buf),  &H_buf));
        CL_CHECK(clSetKernelArg(sw_kern, 1, sizeof(q_buf),  &q_buf));
        CL_CHECK(clSetKernelArg(sw_kern, 2, sizeof(r_buf),  &r_buf));
        CL_CHECK(clSetKernelArg(sw_kern, 3, sizeof(cl_int), &sw_m));
        CL_CHECK(clSetKernelArg(sw_kern, 4, sizeof(cl_int), &sw_n));
        // arg 5 = diag (below)
        CL_CHECK(clSetKernelArg(sw_kern, 6, sizeof(cl_int), &sw_match));
        CL_CHECK(clSetKernelArg(sw_kern, 7, sizeof(cl_int), &sw_mm));
        CL_CHECK(clSetKernelArg(sw_kern, 8, sizeof(cl_int), &sw_gap));

        // Wavefront loop — collect first and last events for timing
        wave_events.reserve(static_cast<std::size_t>(m + n));

        for (int diag = 2; diag <= m + n; ++diag) {
            int i_min = std::max(1, diag - n);
            int i_max = std::min(m, diag - 1);
            if (i_min > i_max) continue;

            std::size_t global = static_cast<std::size_t>(i_max - i_min + 1);
            cl_int diag_arg = diag;
            CL_CHECK(clSetKernelArg(sw_kern, 5, sizeof(cl_int), &diag_arg));

            cl_event ev;
            CL_CHECK(clEnqueueNDRangeKernel(queue, sw_kern, 1,
                                             nullptr, &global, nullptr, 0, nullptr, &ev));
            wave_events.push_back(ev);
        }
    }

    // find_max — runs after all wavefront kernels complete
    cl_int fm_rows = m + 1, fm_cols = n + 1;
    CL_CHECK(clSetKernelArg(findmax_kern, 0, sizeof(H_buf),    &H_buf));
    CL_CHECK(clSetKernelArg(findmax_kern, 1, sizeof(cl_int),   &fm_rows));
    CL_CHECK(clSetKernelArg(findmax_kern, 2, sizeof(cl_int),   &fm_cols));
    CL_CHECK(clSetKernelArg(findmax_kern, 3, sizeof(score_buf),&score_buf));
    CL_CHECK(clSetKernelArg(findmax_kern, 4, sizeof(row_buf),  &row_buf));
    CL_CHECK(clSetKernelArg(findmax_kern, 5, sizeof(col_buf),  &col_buf));

    std::size_t one = 1;
    cl_event fm_ev;
    CL_CHECK(clEnqueueNDRangeKernel(queue, findmax_kern, 1,
                                     nullptr, &one, nullptr, 0, nullptr, &fm_ev));
    CL_CHECK(clFinish(queue));

    // Read results
    cl_int sw_score = 0, sw_row = 0, sw_col = 0;
    CL_CHECK(clEnqueueReadBuffer(queue, score_buf, CL_TRUE, 0,
                                  sizeof(cl_int), &sw_score, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(queue, row_buf,   CL_TRUE, 0,
                                  sizeof(cl_int), &sw_row,   0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(queue, col_buf,   CL_TRUE, 0,
                                  sizeof(cl_int), &sw_col,   0, nullptr, nullptr));

    // Timing: from start of first wavefront kernel to end of find_max
    cl_ulong t0 = 0, t1 = 0;
    if (!wave_events.empty()) {
        clGetEventProfilingInfo(wave_events.front(),
            CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
    }
    clGetEventProfilingInfo(fm_ev,
        CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
    double sw_ms = static_cast<double>(t1 - t0) / 1.0e6;

    for (auto& ev : wave_events) clReleaseEvent(ev);
    clReleaseEvent(fm_ev);

    std::cout << "=== Smith-Waterman (OpenCL) ===\n";
    std::cout << "Score : " << sw_score << "  (max cell at row "
              << sw_row << ", col " << sw_col << ")\n";
    std::printf("[OPENCL] SW Time: %.3f ms\n\n", sw_ms);

    // ══════════════════════════════════════════════════════════════════════
    // KMP / Parallel pattern search
    // ══════════════════════════════════════════════════════════════════════

    cl_program kmp_prog  = build_program(ctx, device, kmp_src);
    cl_kernel  kmp_kern  = clCreateKernel(kmp_prog, "naive_search",  &err); CL_CHECK(err);
    cl_kernel  cnt_kern  = clCreateKernel(kmp_prog, "count_matches", &err); CL_CHECK(err);

    // Determine pattern length (default to min(query_len, 50) if not specified)
    std::size_t max_pattern_len = std::min<std::size_t>(query.size(), 50);
    if (pattern_length > 0) {
        max_pattern_len = static_cast<std::size_t>(pattern_length);
    }
    
    // Validate pattern length
    if (max_pattern_len > query.size()) {
        std::cerr << "Warning: pattern_length (" << max_pattern_len 
                  << ") exceeds query length (" << query.size() 
                  << "), using query length\n";
        max_pattern_len = query.size();
    }
    if (max_pattern_len > ref.size()) {
        std::cerr << "Warning: pattern_length (" << max_pattern_len 
                  << ") exceeds reference length (" << ref.size() 
                  << "), using reference length\n";
        max_pattern_len = ref.size();
    }
    
    const std::string pattern = query.substr(0, max_pattern_len);
    cl_int p_len = static_cast<int>(pattern.size());
    cl_int t_len = static_cast<int>(ref.size());

    std::vector<cl_int> matches_host(static_cast<std::size_t>(t_len), 0);
    cl_mem txt_buf = clCreateBuffer(ctx,
        CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
        static_cast<std::size_t>(t_len),
        const_cast<char*>(ref.data()), &err); CL_CHECK(err);
    cl_mem pat_buf = clCreateBuffer(ctx,
        CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
        static_cast<std::size_t>(p_len),
        const_cast<char*>(pattern.data()), &err); CL_CHECK(err);
    cl_mem mat_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        static_cast<std::size_t>(t_len) * sizeof(cl_int), nullptr, &err); CL_CHECK(err);
    
    // Initialize total count to 0 before atomic adds
    cl_int zero = 0;
    cl_mem tot_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_int), &zero, &err); CL_CHECK(err);

    CL_CHECK(clSetKernelArg(kmp_kern, 0, sizeof(txt_buf), &txt_buf));
    CL_CHECK(clSetKernelArg(kmp_kern, 1, sizeof(cl_int),  &t_len));
    CL_CHECK(clSetKernelArg(kmp_kern, 2, sizeof(pat_buf), &pat_buf));
    CL_CHECK(clSetKernelArg(kmp_kern, 3, sizeof(cl_int),  &p_len));
    CL_CHECK(clSetKernelArg(kmp_kern, 4, sizeof(mat_buf), &mat_buf));

    // Use optimized work-group size (256 is typically good for most GPUs)
    std::size_t kmp_local = 256;
    std::size_t kmp_global = ((t_len + kmp_local - 1) / kmp_local) * kmp_local;
    
    cl_event kmp_ev, cnt_ev;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kmp_kern, 1,
                                     nullptr, &kmp_global, &kmp_local, 0, nullptr, &kmp_ev));

    // Setup parallel reduction with work-group size of 256
    std::size_t local_size = 256;
    std::size_t global_size = ((t_len + local_size - 1) / local_size) * local_size;
    
    CL_CHECK(clSetKernelArg(cnt_kern, 0, sizeof(mat_buf), &mat_buf));
    CL_CHECK(clSetKernelArg(cnt_kern, 1, sizeof(cl_int),  &t_len));
    CL_CHECK(clSetKernelArg(cnt_kern, 2, sizeof(tot_buf), &tot_buf));
    CL_CHECK(clSetKernelArg(cnt_kern, 3, local_size * sizeof(cl_int), nullptr)); // __local scratch
    
    CL_CHECK(clEnqueueNDRangeKernel(queue, cnt_kern, 1,
                                     nullptr, &global_size, &local_size, 0, nullptr, &cnt_ev));
    CL_CHECK(clFinish(queue));

    cl_int hit_total = 0;
    CL_CHECK(clEnqueueReadBuffer(queue, tot_buf, CL_TRUE, 0,
                                  sizeof(cl_int), &hit_total, 0, nullptr, nullptr));

    cl_ulong kt0 = 0, kt1 = 0;
    clGetEventProfilingInfo(kmp_ev,
        CL_PROFILING_COMMAND_START, sizeof(kt0), &kt0, nullptr);
    clGetEventProfilingInfo(cnt_ev,
        CL_PROFILING_COMMAND_END,   sizeof(kt1), &kt1, nullptr);
    double kmp_ms = static_cast<double>(kt1 - kt0) / 1.0e6;

    clReleaseEvent(kmp_ev);
    clReleaseEvent(cnt_ev);

    std::cout << "=== Pattern Search (OpenCL) ===\n";
    std::cout << "Pattern  : " << pattern << '\n';
    std::cout << "Hits     : " << hit_total << '\n';
    std::printf("[OPENCL] KMP Time: %.3f ms\n", kmp_ms);

    // ── Cleanup ────────────────────────────────────────────────────────────
    clReleaseMemObject(H_buf);   clReleaseMemObject(q_buf);
    clReleaseMemObject(r_buf);   clReleaseMemObject(score_buf);
    clReleaseMemObject(row_buf); clReleaseMemObject(col_buf);
    clReleaseMemObject(txt_buf); clReleaseMemObject(pat_buf);
    clReleaseMemObject(mat_buf); clReleaseMemObject(tot_buf);
    clReleaseKernel(sw_kern);    clReleaseKernel(findmax_kern);
    clReleaseKernel(kmp_kern);   clReleaseKernel(cnt_kern);
    clReleaseProgram(sw_prog);   clReleaseProgram(kmp_prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
