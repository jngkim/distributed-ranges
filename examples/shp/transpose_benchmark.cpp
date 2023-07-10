// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>
#include <fmt/core.h>
#include <memory>
#include <oneapi/mkl/blas.hpp>

namespace fft {

template<typename T1, typename T2>
sycl::event transpose_mkl(size_t m, size_t n,
                      T1 in,  size_t lda, T2 out, size_t ldb,
                      const std::vector<sycl::event>& events = {})
{
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);
  return oneapi::mkl::blas::row_major::omatcopy(q,oneapi::mkl::transpose::trans, m, n, 1, in.get_raw_pointer(), lda, out.get_raw_pointer(), ldb, events);
}

template<typename T1, typename T2>
sycl::event transpose(size_t m, size_t n,
                      T1 in,  size_t lda, T2 out, size_t ldb,
                      const std::vector<sycl::event>& events = {})
{
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);
#ifdef USE_MKL_TRANSPOSE
  return oneapi::mkl::blas::row_major::omatcopy(q,oneapi::mkl::transpose::trans, m, n, 1, in.get_raw_pointer(), lda, out.get_raw_pointer(), ldb, events);
#else
  constexpr size_t tile_size = 16;
  const size_t m_max         = ((m + tile_size - 1) / tile_size) * tile_size;
  const size_t n_max         = ((n + tile_size - 1) / tile_size) * tile_size;
  using temp_t = std::iter_value_t<T1>;
  const auto in_ = in.get_raw_pointer();

  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(events);
    sycl::local_accessor<temp_t, 2> tile(sycl::range<2>(tile_size, tile_size + 1), cgh);

    cgh.parallel_for(sycl::nd_range<2>{{m_max, n_max}, {tile_size, tile_size}}, [=](sycl::nd_item<2> item) {
      unsigned x   = item.get_global_id(1);
      unsigned y   = item.get_global_id(0);
      unsigned xth = item.get_local_id(1);
      unsigned yth = item.get_local_id(0);

      if (x < n && y < m)
        tile[yth][xth] = in_[(y)*lda + x];
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

  int nreps = 100;
  std::size_t nprocs = shp::nprocs();
  std::size_t m_local = 64; //4*1024;
  if (argc >= 2) {
    m_local = std::atoll(argv[1]);
  }
  if (argc == 3) {
    nreps = std::atoi(argv[2]);
  }
  std::size_t m = nprocs * m_local;
  std::size_t lda = m;
  std::size_t n_elements = m_local * lda;
  std::size_t block_size = m_local * m_local;

  fmt::print("Transfer size {} GB \n", sizeof(T) * n_elements * 1e-9);

  fmt::print("Allocating...\n");
  shp::distributed_vector<T> i_vec(nprocs * n_elements);
  shp::distributed_vector<T> o_vec(nprocs * n_elements);
  //shp::iota(i_vec.begin(), i_vec.end(), 0);
  shp::fill(i_vec.begin(), i_vec.end(), 11.1);
  shp::fill(o_vec.begin(), o_vec.end(), -1);
  std::vector<T> lv(nprocs * n_elements);

  fmt::print("BW tests...\n");
  std::vector<sycl::event> events;
  auto begin = std::chrono::high_resolution_clock::now();
  for(int iter=0; iter < nreps+1; iter++)
  {
    //transpose(A,B)
    for (std::size_t i = 0; i < nprocs; i++)
    {
      for (std::size_t j_ = 0; j_ < nprocs; j_++)
      {
        std::size_t j = (j_ + i) % std::size_t(nprocs);
        auto &&send = i_vec.segments()[i];
        auto &&receive = o_vec.segments()[j];
        auto e = fft::transpose(m_local, m_local, send.begin() + j * m_local, lda, receive.begin() + i * m_local, lda);
        events.push_back(e);
      }
    }
    sycl::event::wait(events);
    events.clear();

    if(iter==0)
      begin = std::chrono::high_resolution_clock::now();
  }

  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  std::size_t n_bytes = 2 * block_size * nprocs * nprocs * sizeof(T) * nreps;
  double n_gbytes = double(n_bytes) * 1e-9;

  double bw = n_gbytes / duration;

  fmt::print("transposition {}x{} on {} devices -> {} GB/s\n", m, m, shp::nprocs(), bw);

  return 0;
}
