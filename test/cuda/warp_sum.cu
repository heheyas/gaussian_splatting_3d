#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

class GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start, 0); }

  void Stop() { cudaEventRecord(stop, 0); }

  void Elapsed(const char *msg) {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("%s: %.3f ms\n", msg, elapsed);
  }
};

__global__ void warpSum(int *data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int value = data[idx];

#pragma unroll
  for (int i = 1; i < warpSize; i *= 2)
    value += __shfl_xor(value, i);

  printf("idx=%d\tvalue=%d\n", idx, value);
}

__global__ void warpScan(int *data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int value = data[idx];

  int laneId = threadIdx.x % warpSize;

#pragma unroll
  for (int i = 1; i < warpSize; i *= 2) {
    int n = __shfl_up(value, i);
    if (laneId >= i)
      value += n;
  }

  printf("idx=%d\tvalue=%d\n", idx, value);
}

int main() {
  int warpSize = 32;

  // warp sum
  thrust::device_vector<int> A(warpSize);
  thrust::sequence(A.data(), A.data() + warpSize);
  GpuTimer timer;
  timer.Start();
  warpSum<<<1, warpSize>>>(thrust::raw_pointer_cast(A.data()));
  timer.Stop();
  timer.Elapsed("warpSum");

  // warp scan
  thrust::device_vector<int> B(warpSize);
  thrust::sequence(B.data(), B.data() + warpSize);
  warpScan<<<1, warpSize>>>(thrust::raw_pointer_cast(B.data()));

  cudaDeviceSynchronize();
  return 0;
}