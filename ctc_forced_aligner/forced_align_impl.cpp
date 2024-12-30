#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Inspired from
// https://github.com/flashlight/sequence/blob/main/flashlight/lib/sequence/criterion/cpu/ConnectionistTemporalClassificationCriterion.cpp
template <typename scalar_t, typename target_t>
void forced_align_impl(
    const py::array_t<scalar_t>& logProbs,
    const py::array_t<target_t>& targets,
    const int64_t blank,
    py::array_t<target_t>& paths) {
  const scalar_t kNegInfinity = -std::numeric_limits<scalar_t>::infinity();
  const auto batchIndex =
      0; // TODO: support batch version and use the real batch index
  const auto T = logProbs.shape(1);
  const auto L = targets.shape(1);
  const auto S = 2 * L + 1;
  std::vector<scalar_t> alphas(2 * S, kNegInfinity);

  // Replace backPtr tensor with two std::vector<bool>
  // allocate memory based on the expected needed size which is approximately
  // S * (T-L), we will use a safety margin of (T-L) to avoid reallocation
  std::vector<unsigned long long> backPtr_offset(T - 1);
  std::vector<unsigned long long> backPtr_seek(T - 1);

  auto logProbs_data = logProbs.template unchecked<3>();
  auto targets_data = targets.template unchecked<2>();
  auto paths_data = paths.template mutable_unchecked<2>();

  auto R = 0;
  for (auto i = 1; i < L; i++) {
    if (targets_data(batchIndex, i) == targets_data(batchIndex, i - 1)) {
      ++R;
    }
  }

  if (T < L + R) {
      throw std::runtime_error("targets length is too long for CTC.");
  }
  std::vector<bool> backPtrBit0((S + 1) * (T - L), false);
  std::vector<bool> backPtrBit1((S + 1) * (T - L), false);

  auto start = T - (L + R) > 0 ? 0 : 1;
  auto end = (S == 1) ? 1 : 2;
  for (auto i = start; i < end; i++) {
    auto labelIdx = (i % 2 == 0) ? blank : targets_data(batchIndex, i / 2);
    alphas[i] = logProbs_data(batchIndex, 0, labelIdx);
  }
  unsigned long long seek = 0;
  for (auto t = 1; t < T; t++) {
    if (T - t <= L + R) {
      if ((start % 2 == 1) &&
        targets_data(batchIndex, start / 2) != targets_data(batchIndex, start / 2 + 1)) {
        start = start + 1;
      }
      start = start + 1;
    }
    if (t <= L + R) {
      if (end % 2 == 0 && end < 2 * L &&
        targets_data(batchIndex, end / 2 - 1) != targets_data(batchIndex, end / 2)) {
        end = end + 1;
      }
      end = end + 1;
    }
    auto startloop = start;
    auto curIdxOffset = t % 2;
    auto prevIdxOffset = (t - 1) % 2;
    std::fill(alphas.begin() + curIdxOffset * S, alphas.begin() + (curIdxOffset + 1) * S, kNegInfinity);
    backPtr_seek[t - 1] = seek;
    backPtr_offset[t - 1] = start;
    if (start == 0) {
      alphas[curIdxOffset * S] = alphas[prevIdxOffset * S] + logProbs_data(batchIndex, t, blank);
      startloop += 1;
      seek += 1;
    }

    for (auto i = startloop; i < end; i++) {
      auto x0 = alphas[prevIdxOffset * S + i];
      auto x1 = alphas[prevIdxOffset * S + i - 1];
      auto x2 = kNegInfinity;

      auto labelIdx = (i % 2 == 0) ? blank : targets_data(batchIndex, i / 2);

      // In CTC, the optimal path may optionally chose to skip a blank label.
      // x2 represents skipping a letter, and can only happen if we're not
      // currently on a blank_label, and we're not on a repeat letter
      // (i != 1) just ensures we don't access targets[i - 2] if its i < 2
      if (i % 2 != 0 && i != 1 &&
        targets_data(batchIndex, i / 2) != targets_data(batchIndex, i / 2 - 1)) {
        x2 = alphas[prevIdxOffset * S + i - 2];
      }
      scalar_t result = 0.0;
      if (x2 > x1 && x2 > x0) {
        result = x2;
        backPtrBit1[seek + i - startloop] = true;
      } else if (x1 > x0 && x1 > x2) {
        result = x1;
        backPtrBit0[seek + i - startloop] = true;
      } else {
        result = x0;
      }
      alphas[curIdxOffset * S + i] = result + logProbs_data(batchIndex, t, labelIdx);
    }
    seek += (end - startloop);
  }
  auto idx1 = (T - 1) % 2;
  auto ltrIdx = alphas[idx1 * S + S - 1] > alphas[idx1 * S + S - 2] ? S - 1 : S - 2;
  // path stores the token index for each time step after force alignment.
  for (auto t = T - 1; t > -1; t--) {
    auto lbl_idx = ltrIdx % 2 == 0 ? blank : targets_data(batchIndex, ltrIdx / 2);
    paths_data(batchIndex, t) = lbl_idx;
    // Calculate backPtr value from bits
    auto t_minus_one = t - 1 >= 0 ? t - 1 : 0;
    auto backPtr_idx = backPtr_seek[t_minus_one] +
                       ltrIdx - backPtr_offset[t_minus_one];
    ltrIdx -= (backPtrBit1[backPtr_idx] << 1) | backPtrBit0[backPtr_idx];
  }
}

std::tuple<py::array_t<int64_t>, py::array_t<float>> compute(
    const py::array_t<float>& logProbs,
    const py::array_t<int64_t>& targets,
    const int64_t blank) {

  if (logProbs.ndim() != 3) throw std::runtime_error("log_probs must be a 3-D array.");
  if (targets.ndim() != 2) throw std::runtime_error("targets must be a 2-D array.");
  if (logProbs.shape(0) != 1) throw std::runtime_error("Batch size must be 1.");

  const auto B = logProbs.shape(0);
  const auto T = logProbs.shape(1);
  auto paths = py::array_t<int64_t>({B, T});

  forced_align_impl<float, int64_t>(logProbs, targets, blank, paths);

  auto aligned_paths = paths.unchecked<2>();
  auto scores = py::array_t<float>({T});
  auto scores_data = scores.mutable_data();

  auto logProbs_data = logProbs.unchecked<3>();
  for (auto t = 0; t < T; ++t) {
      scores_data[t] = logProbs_data(0, t, aligned_paths(0, t));
  }

  return std::make_tuple(paths, scores);
}

PYBIND11_MODULE(ctc_forced_aligner, m) {
    m.def("forced_align", &compute, "Compute forced alignment.");
}