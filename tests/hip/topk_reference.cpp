// Local experiment: independent CPU oracle for selection, reference mass and
// compact output strides, including ties and extreme finite logits.
#include <hip/hip_runtime.h>
#include "strata/kernels/router_top10.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#define HIP_OK(call) do { auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

int main(int argc, char** argv) {
    const int k = argc > 1 ? std::atoi(argv[1]) : 10;
    const int reference = argc > 2 ? std::atoi(argv[2]) : k;
    constexpr int rows = 129, experts = 512, guard = 16;
    std::vector<float> logits(rows * experts);
    std::mt19937 rng(75431);
    std::normal_distribution<float> normal(0.f, 2.f);
    for (int r = 0; r < rows; ++r)
        for (int e = 0; e < experts; ++e) {
            float x = normal(rng);
            if (r % 4 == 0) x = std::round(x * 2.f) / 2.f;
            if (r % 4 == 1) x *= 40.f;
            if (r % 4 == 2) x = 1.f;
            logits[r * experts + e] = x;
        }
    float *dl = nullptr, *dw = nullptr;
    int* di = nullptr;
    HIP_OK(hipMalloc(&dl, logits.size() * sizeof(float)));
    HIP_OK(hipMalloc(&dw, (rows * k + guard) * sizeof(float)));
    HIP_OK(hipMalloc(&di, (rows * k + guard) * sizeof(int)));
    HIP_OK(hipMemcpy(dl, logits.data(), logits.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(dw, 0x3c, (rows * k + guard) * 4));
    HIP_OK(hipMemset(di, 0x11, (rows * k + guard) * 4));
    if (!strata::kernels::router_top10_variant(dl, rows, experts, k, di, dw, nullptr, 1)) return 3;
    HIP_OK(hipDeviceSynchronize());
    std::vector<int> ids(rows * k + guard);
    std::vector<float> weights(rows * k + guard);
    HIP_OK(hipMemcpy(ids.data(), di, ids.size() * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(weights.data(), dw, weights.size() * 4, hipMemcpyDeviceToHost));
    int failed = 0;
    double max_error = 0;
    for (int r = 0; r < rows; ++r) {
        const float* l = logits.data() + r * experts;
        const float maximum = *std::max_element(l, l + experts);
        std::vector<double> ex(experts);
        double total = 0;
        for (int e = 0; e < experts; ++e) total += ex[e] = std::exp(double(l[e]) - maximum);
        const float inv = float(1. / total);
        std::vector<float> p(experts);
        std::vector<int> order(experts);
        std::iota(order.begin(), order.end(), 0);
        for (int e = 0; e < experts; ++e) p[e] = float(ex[e] * inv);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
        double sum = 0;
        for (int rank = 0; rank < reference; ++rank) sum += p[order[rank]];
        sum = std::max(sum, 6.103515625e-5);
        for (int rank = 0; rank < k; ++rank) {
            const double expected = float(double(p[order[rank]]) / sum);
            const double error = std::abs(double(weights[r * k + rank]) - expected);
            max_error = std::max(max_error, error);
            failed += ids[r * k + rank] != order[rank] || error > 2e-6;
        }
    }
    for (int i = rows * k; i < rows * k + guard; ++i) {
        failed += ids[i] != 0x11111111;
        const unsigned* bits = reinterpret_cast<const unsigned*>(&weights[i]);
        failed += *bits != 0x3c3c3c3c;
    }
    HIP_OK(hipFree(dl)); HIP_OK(hipFree(di)); HIP_OK(hipFree(dw));
    std::printf("topk_reference K=%d reference=%d rows=%d max_abs_error=%.9g failures=%d\n",
                k, reference, rows, max_error, failed);
    return failed ? 1 : 0;
}
