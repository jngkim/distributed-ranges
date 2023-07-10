// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <random>

#include <sycl/sycl.hpp>

#include "cxxopts.hpp"

#include <dr/mhp.hpp>
#include <dr/shp.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

namespace mhp = dr::mhp;

using T = double;

MPI_Comm comm;
std::size_t comm_rank;
std::size_t comm_size;

std::size_t n;
std::size_t n_iterations;


auto Stream_Copy(dr::distributed_range auto &&a,
                             dr::distributed_range auto &&b) {
  mhp::for_each(mhp::views::zip(a, b), [](auto &&v) { std::get<1>(v) = std::get<0>(v); });
}

auto Stream_Scale(dr::distributed_range auto &&a,
                             dr::distributed_range auto &&b) {
  T scalar = 45.96f;
  mhp::for_each(mhp::views::zip(a, b), [scalar](auto &&v) { std::get<1>(v) = scalar * std::get<0>(v); });
}

auto Stream_Add(dr::distributed_range auto &&a,
                             dr::distributed_range auto &&b, dr::distributed_range auto &&c) {
  mhp::for_each(mhp::views::zip(a, b, c), [](auto &&v) { std::get<2>(v) = std::get<0>(v) + std::get<1>(v); });
}

auto Stream_Triad(dr::distributed_range auto &&a,
                             dr::distributed_range auto &&b, dr::distributed_range auto &&c) {
  T scalar = 45.96f;
  mhp::for_each(mhp::views::zip(a, b), [scalar](auto &&v) { std::get<1>(v) = std::get<0>(v) + scalar * std::get<1>(v); });
}


int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  comm_rank = rank;
  comm_size = size;

  cxxopts::Options options_spec(argv[0], "mhp dot product benchmark");
  // clang-format off
  options_spec.add_options()
    ("help", "Print help")
    ("i", "Number of iterations", cxxopts::value<std::size_t>()->default_value("10"))
    ("log", "Enable logging")
    ("n", "Size of array", cxxopts::value<std::size_t>()->default_value("1000000"))
    ("sycl", "Execute on sycl device");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  if (options.count("help")) {
    std::cout << options_spec.help() << "\n";
    exit(0);
  }

  std::ofstream *logfile = nullptr;
  if (options.count("log")) {
    logfile = new std::ofstream(fmt::format("dr.{}.log", comm_rank));
    dr::drlog.set_file(*logfile);
  }
  dr::drlog.debug("Rank: {}\n", comm_rank);

  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  sycl::queue q{devices[comm_rank%devices.size()]}; //cyclic

  if (options.count("sycl")) {
    mhp::init(q);
  } else {
    mhp::init();
  }

  n = options["n"].as<std::size_t>();
  n_iterations = options["i"].as<std::size_t>();
  // std::vector<T> a_local(n);
  // std::vector<T> b_local(n);
  // std::vector<T> c_local(n);

  if (comm_rank == 0) {
    fmt::print("Initializing vectors...\n");
  }
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<> dist(0, 1000);
  // for (std::size_t i = 0; i < n; i++) {
  //   a_local[i] = dist(rng);
  //   b_local[i] = dist(rng);
  //   c_local[i] = dist(rng);
  // }

  mhp::distributed_vector<T> a(n, 5.0f);
  mhp::distributed_vector<T> b(n, 5.0f);
  mhp::distributed_vector<T> c(n, 5.0f);
  // mhp::copy(0, a_local, a.begin());
  // mhp::copy(0, b_local, b.begin());
  // mhp::copy(0, c_local, c.begin());

  std::vector<double> durations;
  durations.reserve(n_iterations);

  if (comm_rank == 0) {
    fmt::print("Vectors initialized, starting benchmark...\n");
  }

  // Execute on all devices with MHP:
  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    Stream_Copy(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }
  if (comm_rank == 0) {
    auto avg_time = std::reduce(durations.begin(), durations.end())/durations.size();
    auto bandwidth = (2 * n * sizeof(T)) / (avg_time * 1024 * 1024 * 1024 );
    fmt::print("Stream_Copy_avg_time_ms\t{:.6}\tStream_Copy_bandwidth_GBps\t{:.6}\n", avg_time, bandwidth);
  }


  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    Stream_Scale(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }
  if (comm_rank == 0) {
    auto avg_time = std::reduce(durations.begin(), durations.end())/durations.size();
    auto bandwidth = (2 * n * sizeof(T)) / (avg_time * 1024 * 1024 * 1024 );
    fmt::print("Stream_Scale_avg_time_ms\t{:.6}\tStream_Scale_bandwidth_GBps\t{:.6}\n", avg_time, bandwidth);
  }


  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    Stream_Add(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }
  if (comm_rank == 0) {
    auto avg_time = std::reduce(durations.begin(), durations.end())/durations.size();
    auto bandwidth = (3 * n * sizeof(T)) / (avg_time * 1024 * 1024 * 1024 );
    fmt::print("Stream_Add_avg_time_ms\t{:.6}\tStream_Add_bandwidth_GBps\t{:.6}\n", avg_time, bandwidth);
  }


  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    Stream_Triad(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }
  if (comm_rank == 0) {
    auto avg_time = std::reduce(durations.begin(), durations.end())/durations.size();
    auto bandwidth = (3 * n * sizeof(T)) / (avg_time * 1024 * 1024 * 1024 );
    fmt::print("Stream_Triad_avg_time_ms\t{:.6}\tStream_Triad_bandwidth_GBps\t{:.6}\n", avg_time, bandwidth);
  }

  return 0;
}
