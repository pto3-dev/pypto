/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "pypto/ir/transforms/dsa/dsa_reuse_penalty_solver.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

namespace dsa = pypto::ir::dsa;

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

const dsa::DsaSolution& RequireSolution(const dsa::DsaResult& result, const std::string& message) {
  if (!result.solution.has_value()) throw std::runtime_error(message);
  return result.solution.value();
}

uint64_t OffsetOf(const dsa::DsaSolution& solution, dsa::BufferId id) {
  const uint64_t* offset = solution.Find(id);
  Require(offset != nullptr, "expected buffer " + std::to_string(id) + " to be placed");
  return *offset;
}

void TestFeasiblePackingAndTemporalConflict() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 4}},
      {1, 64, 32, 0, {2, 6}},
      {2, 64, 32, 0, {6, 8}},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "expected a feasible temporal packing");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 0) !=
              OffsetOf(RequireSolution(result, "expected solution"), 1),
          "lifetime-overlapping buffers must be disjoint");
  Require(result.objective.max_peak <= 128, "feasible placement must respect capacity");
  Require(dsa::ValidateSolution(problem, RequireSolution(result, "expected solution")).empty(),
          "solver result must pass independent validation");
}

void TestNoFitDoesNotProveInfeasibility() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 16, {}}};
  problem.buffers = {
      {0, 6, 2, 0, {3, 5}}, {1, 6, 2, 0, {2, 7}}, {2, 4, 4, 0, {4, 5}},
      {3, 8, 4, 0, {0, 3}}, {4, 2, 2, 0, {5, 7}}, {5, 1, 1, 0, {5, 10}},
  };

  // Independently verified strict witness. Canonical greedy's bounded order
  // set misses it; kNoFit therefore means search failure, not infeasibility.
  dsa::DsaSolution witness;
  witness.offsets = {{0, 4}, {1, 10}, {2, 0}, {3, 0}, {4, 0}, {5, 2}};
  Require(dsa::ValidateSolution(problem, witness).empty(),
          "strict witness must prove that the characterization instance fits");

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kNoFit && !result.solution.has_value(),
          "bounded canonical greedy is expected to miss the strict witness");
}

void TestExplicitHardSeparation() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 2}},
      {1, 64, 32, 0, {2, 4}},
  };
  problem.separations = {{0, 1}};

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "expected separated lifetime-disjoint buffers to fit");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 0) !=
              OffsetOf(RequireSolution(result, "expected solution"), 1),
          "explicit separation must prevent address reuse");
}

void TestWeightedPenaltyAvoidance() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 2}},
      {1, 64, 32, 0, {2, 4}},
      {2, 64, 32, 0, {4, 6}},
  };
  problem.reuse_penalties = {
      {0, 1, 10},
      {0, 2, 1},
      {1, 2, 1},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "expected weighted instance to fit");
  Require(result.objective.reuse_cost == 1, "canonical greedy must avoid the weight-10 overlap");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 0) !=
              OffsetOf(RequireSolution(result, "expected solution"), 1),
          "the highest-weight pair should be separated");
  Require(dsa::EvaluateObjective(problem, RequireSolution(result, "expected solution")).reuse_cost == 1,
          "independent objective evaluation must match the solver result");
}

void TestCapacityNoFit() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 4}},
      {1, 64, 32, 0, {0, 4}},
      {2, 64, 32, 0, {0, 4}},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kNoFit && !result.solution.has_value(),
          "three co-live 64-byte buffers cannot fit in 128 bytes");
}

void TestAlignmentAndReservedRanges() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 256, {{0, 32}, {96, 128}}}};
  problem.buffers = {
      {0, 48, 32, 0, {0, 2}},
      {1, 64, 64, 0, {0, 2}},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "expected aligned placement around reserved ranges");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 0) % 32 == 0,
          "buffer 0 must satisfy 32-byte alignment");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 1) % 64 == 0,
          "buffer 1 must satisfy 64-byte alignment");
  Require(dsa::ValidateSolution(problem, RequireSolution(result, "expected solution")).empty(),
          "aligned placement must avoid both reserved ranges");
}

void TestDeterministicResult() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 192, {}}};
  problem.buffers = {
      {7, 64, 32, 0, {0, 2}},
      {3, 64, 32, 0, {2, 4}},
      {9, 64, 32, 0, {4, 6}},
      {1, 64, 32, 0, {6, 8}},
  };
  problem.reuse_penalties = {
      {7, 3, 5},
      {7, 9, 2},
      {3, 1, 4},
      {9, 1, 1},
  };
  const dsa::CanonicalGreedyOptions options{17, 5};

  const dsa::DsaResult first = dsa::CanonicalGreedySolver(options).Solve(problem);
  const dsa::DsaResult second = dsa::CanonicalGreedySolver(options).Solve(problem);
  Require(first.status == dsa::SolveStatus::kFeasible && first.solution.has_value(),
          "first deterministic solve must be feasible");
  Require(second.status == dsa::SolveStatus::kFeasible && second.solution.has_value(),
          "second deterministic solve must be feasible");
  Require(RequireSolution(first, "expected first solution").offsets ==
              RequireSolution(second, "expected second solution").offsets,
          "same seed and problem must produce identical offsets");
  Require(first.objective.reuse_cost == second.objective.reuse_cost,
          "same seed and problem must produce identical objective");
  Require(first.statistics.candidate_offsets_evaluated == second.statistics.candidate_offsets_evaluated,
          "deterministic solve must evaluate the same number of candidates");
}

void TestSecondaryPeakObjective() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 13, {}}};
  problem.buffers = {
      {0, 5, 1, 0, {4, 8}},
      {1, 5, 1, 0, {5, 8}},
      {2, 2, 1, 0, {5, 8}},
      {3, 6, 1, 0, {3, 5}},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "secondary-objective instance must fit");
  Require(result.objective.reuse_cost == 0, "instance has no reuse penalties");
  Require(result.objective.total_peak == 12 && result.objective.max_peak == 12,
          "canonical greedy must retain the lowest peak found on a reuse-cost tie");
}

void TestRejectsPenaltyOnLifetimeConflict() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 4}},
      {1, 64, 32, 0, {2, 6}},
  };
  problem.reuse_penalties = {{0, 1, 1}};

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kInvalidProblem && !result.solution.has_value(),
          "soft edges on lifetime-conflicting buffers must be rejected");
}

void TestPipelineOnlyRelaxation() {
  dsa::DsaProblem strict;
  strict.pools = {{0, 128, {}}};
  strict.buffers = {
      {0, 64, 32, 0, {0, 2}},
      {1, 64, 32, 0, {2, 4}},
      {2, 64, 32, 0, {3, 5}},
  };
  strict.separations = {{0, 1}, {0, 2}};

  const dsa::DsaResult strict_result = dsa::CanonicalGreedySolver().Solve(strict);
  Require(strict_result.status == dsa::SolveStatus::kNoFit,
          "three pairwise-separated buffers cannot fit in two slots");

  const dsa::DsaProblem relaxed = dsa::RelaxSeparationsToPenalties(strict, {{0, 1}}, 1);
  Require(relaxed.separations.size() == 1 && relaxed.separations[0].first == 0 &&
              relaxed.separations[0].second == 2,
          "only the selected pipeline relation may be relaxed");
  Require(relaxed.reuse_penalties.size() == 1 && relaxed.reuse_penalties[0].first == 0 &&
              relaxed.reuse_penalties[0].second == 1 && relaxed.reuse_penalties[0].weight == 1,
          "the relaxed pipeline relation must become a unit penalty");

  const dsa::DsaResult relaxed_result = dsa::CanonicalGreedySolver().Solve(relaxed);
  Require(relaxed_result.status == dsa::SolveStatus::kFeasible && relaxed_result.solution.has_value(),
          "pipeline-only relaxation must recover a capacity-fitting placement");
  Require(
      dsa::CountOverlappingPairs(strict, RequireSolution(relaxed_result, "expected solution"), {{0, 1}}) == 1,
      "the selected fallback activates exactly the relaxed pair");
  Require(dsa::ValidateSolution(relaxed, RequireSolution(relaxed_result, "expected solution")).empty(),
          "the relaxed placement must retain every non-pipeline hard constraint");
}

void TestRelaxedPairCanRemainInactive() {
  dsa::DsaProblem strict;
  strict.pools = {{0, 128, {}}};
  strict.buffers = {
      {0, 64, 32, 0, {0, 2}},
      {1, 64, 32, 0, {2, 4}},
  };
  strict.separations = {{0, 1}};
  const dsa::DsaProblem relaxed = dsa::RelaxSeparationsToPenalties(strict, {{0, 1}}, 1);
  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(relaxed);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "relaxed two-buffer problem must fit");
  Require(result.objective.reuse_cost == 0 &&
              dsa::CountOverlappingPairs(strict, RequireSolution(result, "expected solution"), {{0, 1}}) == 0,
          "a relaxed relation emits no warning when the selected placement still separates it");
}

void TestPipelineRelaxationIgnoresLifetimeConflict() {
  dsa::DsaProblem strict;
  strict.pools = {{0, 192, {}}};
  strict.buffers = {
      {0, 64, 32, 0, {0, 4}},
      {1, 64, 32, 0, {2, 6}},
      {2, 64, 32, 0, {6, 8}},
  };
  strict.separations = {{0, 1}, {1, 2}};

  const dsa::DsaProblem relaxed = dsa::RelaxSeparationsToPenalties(strict, {{0, 1}, {1, 2}}, 1);
  Require(relaxed.separations.empty(), "selected typed separations must be removed");
  Require(relaxed.reuse_penalties.size() == 1 && relaxed.reuse_penalties[0].first == 1 &&
              relaxed.reuse_penalties[0].second == 2,
          "only the lifetime-compatible pipeline relation may become a soft edge");
  Require(dsa::ValidateProblem(relaxed).empty(),
          "mixed co-live/disjoint pipeline fallback must remain a valid DSA-RP problem");

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(relaxed);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "mixed pipeline fallback must remain solvable");
  Require(OffsetOf(RequireSolution(result, "expected solution"), 0) !=
              OffsetOf(RequireSolution(result, "expected solution"), 1),
          "ordinary lifetime interference must remain hard after typed relaxation");
}

void TestIndependentValidationRejectsCorruption() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {
      {0, 64, 32, 0, {0, 4}},
      {1, 64, 32, 0, {0, 4}},
  };

  const dsa::DsaResult result = dsa::CanonicalGreedySolver().Solve(problem);
  Require(result.status == dsa::SolveStatus::kFeasible && result.solution.has_value(),
          "expected a valid baseline placement");

  dsa::DsaSolution corrupted = RequireSolution(result, "expected solution");
  corrupted.offsets[1] = corrupted.offsets[0];
  Require(!dsa::ValidateSolution(problem, corrupted).empty(),
          "independent validation must reject an overlapping hard conflict");

  corrupted = RequireSolution(result, "expected solution");
  corrupted.offsets[0] = 1;
  Require(!dsa::ValidateSolution(problem, corrupted).empty(),
          "independent validation must reject misalignment");
}

void TestReuseBoundaryValidation() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 20000, {}}};
  problem.buffers = {{0, 7168, 32, 0, {0, 2}}, {1, 7168, 32, 0, {2, 4}}};
  problem.reuse_boundaries = {{0, 1, 0, 0}};
  // The YOLO convolution failure wrote the first 64 rows over unread row 64.
  // Temporal compatibility must not permit this shifted physical overlap.
  dsa::DsaSolution solution;
  solution.offsets = {{0, 0}, {1, 4096}};
  Require(!dsa::ValidateSolution(problem, solution).empty(), "reject shifted TADD overlap");
  solution.offsets[1] = 0;
  Require(dsa::ValidateSolution(problem, solution).empty(), "retain exact in-place handoff");
  solution.offsets[1] = 7168;
  Require(dsa::ValidateSolution(problem, solution).empty(), "retain disjoint handoff");

  for (const dsa::ReuseBoundary boundary :
       {dsa::ReuseBoundary{0, 9, 0, 0}, {0, 0, 0, 0}, {0, 1, 7168, 0}, {0, 1, 0, 7168}}) {
    problem.reuse_boundaries = {boundary};
    Require(!dsa::ValidateProblem(problem).empty(), "reject malformed reuse-boundary relation");
  }
  problem.reuse_boundaries = {{0, 1, 0, 0}};
  problem.pools.push_back({1, 20000, {}});
  problem.buffers[1].pool = 1;
  Require(!dsa::ValidateProblem(problem).empty(), "reject reuse boundary across pools");
}

void TestViewBoundaryPacking() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 128, {}}};
  problem.buffers = {{0, 128, 32, 0, {0, 2}}, {1, 64, 32, 0, {2, 4}}};
  problem.reuse_boundaries = {{0, 1, 32, 0}};
  const auto result = dsa::CanonicalGreedySolver().Solve(problem);
  const auto& solution = RequireSolution(result, "view in-place handoff must fit one allocation");
  Require(OffsetOf(solution, 0) == 0 && OffsetOf(solution, 1) == 32,
          "different root bases must preserve coincident accessed member addresses");
  Require(result.statistics.first_fit_seed_feasible,
          "penalty-blind first-fit must find the internal coincidence hole too");
  Require(dsa::ValidateSolution(problem, solution).empty(), "validate member-relative placement");

  // Two operands refer to different parts of the same allocation; no output
  // address can coincide with both, so their constraints force disjointness.
  problem.reuse_boundaries.push_back({0, 1, 64, 0});
  problem.pools[0].capacity = 192;
  const auto distinct = dsa::CanonicalGreedySolver().Solve(problem);
  const auto& separate = RequireSolution(distinct, "incompatible view origins must fit disjointly");
  const auto first = OffsetOf(separate, 0);
  const auto second = OffsetOf(separate, 1);
  Require(first + 128 <= second || second + 64 <= first, "multiple origins require disjointness");
  Require(distinct.statistics.first_fit_seed_feasible, "first-fit must intersect all origin constraints");

  problem.reuse_boundaries = {{0, 1, 1, 0}};
  const auto unaligned = dsa::CanonicalGreedySolver().Solve(problem);
  Require(dsa::ValidateSolution(problem, RequireSolution(unaligned, "unaligned coincidence must separate"))
              .empty(),
          "alignment cannot round an exact coincidence into a shifted overlap");
}

void TestReuseBoundaryReservedAndHardConstraints() {
  dsa::DsaProblem problem;
  problem.pools = {{0, 192, {{0, 32}}}};
  problem.buffers = {{0, 128, 32, 0, {0, 2}}, {1, 64, 32, 0, {2, 4}}};
  problem.reuse_boundaries = {{0, 1, 32, 0}};
  const auto result = dsa::CanonicalGreedySolver().Solve(problem);
  const auto& solution = RequireSolution(result, "reserved window and coincident member must fit");
  Require(OffsetOf(solution, 0) == 32 && OffsetOf(solution, 1) == 64,
          "reserved bytes remain unavailable to both allocations");

  problem.pools[0].capacity = 224;
  problem.separations = {{0, 1}};
  const auto hard = dsa::CanonicalGreedySolver().Solve(problem);
  const auto& separated = RequireSolution(hard, "hard-separated boundary pair must fit");
  Require(OffsetOf(separated, 0) + 128 <= OffsetOf(separated, 1) ||
              OffsetOf(separated, 1) + 64 <= OffsetOf(separated, 0),
          "hard separation takes precedence over exact coincidence");
  const auto relaxed = dsa::RelaxSeparationsToPenalties(problem, {{0, 1}}, 1);
  Require(relaxed.reuse_boundaries.size() == 1, "pipeline fallback retains boundary correctness");
  dsa::DsaSolution shifted;
  shifted.offsets = {{0, 32}, {1, 96}};
  Require(!dsa::ValidateSolution(relaxed, shifted).empty(), "relaxation cannot legalize shifted overlap");

  problem.separations.clear();
  problem.buffers[1].lifetime = {1, 4};
  const auto co_live = dsa::CanonicalGreedySolver().Solve(problem);
  Require(dsa::ValidateSolution(problem, RequireSolution(co_live, "co-live pair must separate")).empty(),
          "ordinary liveness interference remains hard");
}

void TestReuseBoundarySmallGeometryOracle() {
  // Enumerate all byte placements independently of the solver's interval sweep.
  // This covers unequal sizes, holes at either edge, and nonzero view origins.
  for (uint64_t first_size = 1; first_size <= 5; ++first_size) {
    for (uint64_t second_size = 1; second_size <= 5; ++second_size) {
      for (uint64_t origin = 0; origin < first_size; ++origin) {
        dsa::DsaProblem problem;
        problem.pools = {{0, 16, {}}};
        problem.buffers = {{0, first_size, 1, 0, {0, 2}}, {1, second_size, 1, 0, {2, 4}}};
        problem.reuse_boundaries = {{0, 1, origin, 0}};
        for (uint64_t first = 0; first + first_size <= 16; ++first) {
          for (uint64_t second = 0; second + second_size <= 16; ++second) {
            const bool allowed =
                first + first_size <= second || second + second_size <= first || first + origin == second;
            dsa::DsaSolution witness;
            witness.offsets = {{0, first}, {1, second}};
            Require(dsa::ValidateSolution(problem, witness).empty() == allowed,
                    "independent byte geometry disagrees with boundary validation");
          }
        }
        const auto result = dsa::CanonicalGreedySolver().Solve(problem);
        Require(dsa::ValidateSolution(problem, RequireSolution(result, "small boundary geometry must fit"))
                    .empty(),
                "all constructive orders must respect member-origin geometry");
        Require(result.statistics.first_fit_seed_feasible, "small geometry must retain feasible first-fit");
      }
    }
  }
}

void TestReuseBoundaryNearAddressLimit() {
  const uint64_t limit = std::numeric_limits<uint64_t>::max();
  dsa::DsaProblem problem;
  problem.pools = {{0, limit, {{0, limit - 64}}}};
  problem.buffers = {{0, 64, 1, 0, {0, 2}}, {1, 32, 1, 0, {2, 4}}};
  problem.reuse_boundaries = {{0, 1, 32, 0}};
  const auto result = dsa::CanonicalGreedySolver().Solve(problem);
  const auto& solution = RequireSolution(result, "highest valid member coincidence must not overflow");
  Require(OffsetOf(solution, 0) == limit - 64 && OffsetOf(solution, 1) == limit - 32,
          "near-limit address arithmetic must remain exact");
  Require(dsa::ValidateSolution(problem, solution).empty(), "validate near-limit solution");
}

}  // namespace

int main() {
  try {
    TestFeasiblePackingAndTemporalConflict();
    TestNoFitDoesNotProveInfeasibility();
    TestExplicitHardSeparation();
    TestWeightedPenaltyAvoidance();
    TestCapacityNoFit();
    TestAlignmentAndReservedRanges();
    TestDeterministicResult();
    TestSecondaryPeakObjective();
    TestRejectsPenaltyOnLifetimeConflict();
    TestPipelineOnlyRelaxation();
    TestRelaxedPairCanRemainInactive();
    TestPipelineRelaxationIgnoresLifetimeConflict();
    TestIndependentValidationRejectsCorruption();
    TestReuseBoundaryValidation();
    TestViewBoundaryPacking();
    TestReuseBoundaryReservedAndHardConstraints();
    TestReuseBoundarySmallGeometryOracle();
    TestReuseBoundaryNearAddressLimit();
  } catch (const std::exception& error) {
    std::cerr << "DSA-RP solver test failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
