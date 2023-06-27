// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>
#include <fmt/core.h>
#include <memory>
#include <oneapi/mkl/blas.hpp>

#define USE_MKL

namespace tryme {

template<typename T1, typename T2>
sycl::event transpose(size_t m, size_t n,
                      T1 in,  size_t lda, T2 out, size_t ldb,
                      const std::vector<sycl::event>& events = {})
{
  //gather mode
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);

#ifdef USE_MKL
  return oneapi::mkl::blas::row_major::omatcopy(q,oneapi::mkl::transpose::trans, m, n, 1, in.get_raw_pointer(), lda, out.get_raw_pointer(), ldb, events);
#else
  constexpr size_t tile_size = 16;
  const size_t m_max         = ((m + tile_size - 1) / tile_size) * tile_size;
  const size_t n_max         = ((n + tile_size - 1) / tile_size) * tile_size;
  using temp_t = std::iter_value_t<T1>;

  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(events);
    sycl::local_accessor<temp_t, 2> tile(sycl::range<2>(tile_size, tile_size + 1), cgh);

    cgh.parallel_for(sycl::nd_range<2>{{m_max, n_max}, {tile_size, tile_size}}, [=](sycl::nd_item<2> item) {
      unsigned x   = item.get_global_id(1);
      unsigned y   = item.get_global_id(0);
      unsigned xth = item.get_local_id(1);
      unsigned yth = item.get_local_id(0);

      if (x < n && y < m)
        tile[yth][xth] = in[(y)*lda + x];
      item.barrier(sycl::access::fence_space::local_space);

      x = item.get_group(0) * tile_size + xth;
      y = item.get_group(1) * tile_size + yth;
      if (x < m && y < n)
        out[(y)*ldb + x] = tile[xth][yth];
    });
  });
#endif
}
}

namespace shp = dr::shp;

int main(int argc, char **argv) {
  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  shp::init(devices);

  using T = float;

  std::size_t nprocs = shp::nprocs();
  std::size_t m_local = 4*1024;
  std::size_t m = nprocs * m_local;
  std::size_t lda = m;
  std::size_t n_elements = m_local * lda;
  std::size_t block_size = m_local * m_local;

  if (argc == 2) {
    n_elements = std::atoll(argv[1]);
  }

  fmt::print("Transfer size {} GB\n", n_elements * 1e-9);

  using vector_type = shp::vector<T, shp::device_allocator<T>>;

  std::vector<vector_type> in_data;
  std::vector<vector_type> out_data;

  fmt::print("Allocating...\n");

  for (std::size_t i = 0; i < shp::nprocs(); i++) {
    shp::device_allocator<T> allocator(shp::context(), shp::devices()[i]);
    in_data.emplace_back(n_elements, allocator);
    out_data.emplace_back(n_elements, allocator);

    shp::fill(in_data.back().begin(), in_data.back().end(), i);
    shp::fill(out_data.back().begin(), out_data.back().end(), -1);
  }

  fmt::print("BW tests...\n");
  int nreps = 1;
  std::vector<sycl::event> events;
  auto begin = std::chrono::high_resolution_clock::now();
  for(int iter=0; iter < nreps; iter++)
  {
    //transpose(A,B); 
    for (std::size_t j = 0; j < nprocs; ++j)
    {
      for (std::size_t i = 0; i < nprocs; ++i)
      {
        auto &&send = in_data[i];
        auto &&receive = out_data[j];
        auto e = tryme::transpose(m_local, m_local, send.begin() + j * m_local, lda, receive.begin() + i * m_local, lda);
        events.push_back(e);
      }
      sycl::event::wait(events);
      events.clear();
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  std::size_t n_bytes = 2 * block_size * nprocs * sizeof(T) * nreps;
  double n_gbytes = double(n_bytes) * 1e-9;

  double bw = n_gbytes / duration;

  fmt::print("transposition {}x{} on {} devices -> {} GB/s\n", m, m, shp::nprocs(), bw);

  return 0;
}
