// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/shp.hpp>
#include <fmt/core.h>
#include <memory>
#include <oneapi/mkl/blas.hpp>

//#define USE_MKL_TRANSPOSE
namespace fft {

template<typename InIT, typename OutIT>
sycl::event transpose(size_t m, size_t n,
                      InIT in,  size_t lda, OutIT out, size_t ldb,
                      const std::vector<sycl::event>& events = {})
{
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);

#ifdef USE_MKL_TRANSPOSE
  return oneapi::mkl::blas::row_major::omatcopy(q,oneapi::mkl::transpose::trans, 
      m, n, 1, in.get_raw_pointer(), lda, out.get_raw_pointer(), ldb, events);
#else
  constexpr size_t tile_size = 16;
  const size_t m_max         = ((m + tile_size - 1) / tile_size) * tile_size;
  const size_t n_max         = ((n + tile_size - 1) / tile_size) * tile_size;
  using temp_t = std::iter_value_t<InIT>;
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

template<typename TileType>
sycl::event transpose_tile(TileType&& send, TileType&& recv, size_t i, size_t j)
{
  auto m_tile = send.shape()[0];
  auto n_tile = recv.shape()[0];
  return transpose(m_tile, n_tile, send.data() + j * n_tile, send.shape()[1],
                   recv.data() + i * m_tile, recv.shape()[1]);
}

}

namespace shp = dr::shp;

int main(int argc, char **argv) {
  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  shp::init(devices);

  using T = float;

  int nreps = 100;
  std::size_t nprocs = shp::nprocs();
  std::size_t m_unit = 64; //4*1024;
  if (argc >= 2) {
    m_unit = std::atoll(argv[1]);
  }
  if (argc == 3) {
    nreps = std::atoi(argv[2]);
  }
  std::size_t n = nprocs * m_unit;
  std::size_t m = 2 * n;

  fmt::print("Transfer size {} GB \n", sizeof(T) * m * n * 1e-9);

  fmt::print("Allocating...\n");
  shp::distributed_dense_matrix<T>  i_mat({m,n},
      dr::shp::block_cyclic({dr::shp::tile::div, dr::shp::tile::div}, {dr::shp::nprocs(), 1}));
  shp::distributed_dense_matrix<T>  o_mat({n,m},
      dr::shp::block_cyclic({dr::shp::tile::div, dr::shp::tile::div}, {dr::shp::nprocs(), 1}));

  //fmt::print("ddm      shape {} {}\n", i_mat.shape()[0], i_mat.shape()[1]);
  //fmt::print("ddm tile_shape {} {}\n", i_mat.tile_shape()[0], i_mat.tile_shape()[1]);
  //fmt::print("ddm grid_shape {} {}\n", i_mat.grid_shape()[0], i_mat.grid_shape()[1]);

  auto shape = i_mat.shape();
  dr::shp::for_each(dr::shp::par_unseq, i_mat, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = static_cast<T>(idx[0]*shape[1] + idx[1]);
  });

  if(m_unit < 16)
    for (auto iter = i_mat.begin(); iter != i_mat.end(); ++iter) {
      auto &&[idx, v] = *iter;
      auto &&[i, j] = idx;
      std::cout << v << " ";
      if(n-j == 1) std::cout << std::endl;
    }

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
        auto &&send = i_mat.tile({i,0});
        auto &&recv = o_mat.tile({j,0});
        auto m_tile = send.shape()[0];
        auto n_tile = recv.shape()[0];
        auto e = fft::transpose(m_tile, n_tile, send.data() + j * n_tile, send.shape()[1], recv.data() + i * m_tile, recv.shape()[1]);
        //auto e = fft::transpose_tile(i_mat.tile({i,0}), o_mat.tile({j,0}), i, j);
        events.push_back(e);
      }
    }
    sycl::event::wait(events);
    events.clear(); if(iter==0)
    {
      if(m_unit<16) 
        for (auto iter = o_mat.begin(); iter != o_mat.end(); ++iter) {
          auto &&[idx, v] = *iter;
          auto &&[i, j] = idx;
          std::cout << v << " ";
          if(m-j == 1) std::cout << std::endl;
        }
      begin = std::chrono::high_resolution_clock::now();
    }

  }

  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  std::size_t n_bytes = 2 * i_mat.size() * sizeof(T) * nreps;
  double n_gbytes = double(n_bytes) * 1e-9;

  double bw = n_gbytes / duration;

  fmt::print("transposition {}x{} on {} devices -> {} GB/s\n", m, m, shp::nprocs(), bw);

  return 0;
}
