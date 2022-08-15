#pragma once
#include <cstring>
#include <vector>

#include "cwl/util.h"

namespace cwl
{

// RAII buffer object which is on device
template <typename T>
class CUDABuffer
{
 public:
  CUDABuffer(uint32_t buffer_size) : m_buffer_size(buffer_size)
  {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_ptr),
                          m_buffer_size * sizeof(T)));
  }

  CUDABuffer(uint32_t buffer_size, int value) : CUDABuffer<T>(buffer_size)
  {
    CUDA_CHECK(cudaMemset(m_d_ptr, value, m_buffer_size * sizeof(T)));
  }

  CUDABuffer(const std::vector<T>& values) : CUDABuffer<T>(values.size())
  {
    copy_from_host_to_device(values);
  }

  CUDABuffer(const CUDABuffer<T>& other) = delete;

  CUDABuffer(CUDABuffer<T>&& other)
      : m_d_ptr(other.m_d_ptr), m_buffer_size(other.m_buffer_size)
  {
    other.m_d_ptr = nullptr;
    other.m_buffer_size = 0;
  }

  ~CUDABuffer() noexcept(false)
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_d_ptr)));
  }

  void clear()
  {
    CUDA_CHECK(cudaMemset(m_d_ptr, 0, m_buffer_size * sizeof(T)));
  }

  void copy_from_host_to_device(const std::vector<T>& value)
  {
    CUDA_CHECK(cudaMemcpy(m_d_ptr, value.data(), m_buffer_size * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  void copy_from_device_to_host(std::vector<T>& value)
  {
    value.resize(m_buffer_size);
    CUDA_CHECK(cudaMemcpy(value.data(), m_d_ptr, m_buffer_size * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  T* get_device_ptr() const { return m_d_ptr; }

  uint32_t get_size() const { return m_buffer_size; }

  uint32_t get_size_in_bytes() const { return m_buffer_size * sizeof(T); }

 private:
  T* m_d_ptr;
  uint32_t m_buffer_size;
};

}  // namespace cwl