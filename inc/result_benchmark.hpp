#ifndef RESULT_BENCHMARK_HPP_
#define RESULT_BENCHMARK_HPP_

#include <array>
#include <vector>
#include <assert.h>

namespace gearshifft
{
/** Result data generated after a benchmark has completed the runs
 */
  template<int T_NumberRuns, int T_NumberValues>
  class ResultBenchmark  {
  public:
    using ValuesT = std::array<std::array<double, T_NumberValues >, T_NumberRuns >;
    using BufferSizesT = std::array<size_t, T_NumberRuns >;

    template<bool isComplex, bool isInplace, size_t T_NDim>
    void init(const std::array<unsigned, T_NDim>& ce) {
      for(auto i=0; i<T_NDim; ++i) {
        extents_[i] = ce[i];
      }
      dim_ = T_NDim;
      dimkind_ = 0; //@todo dimkind
      run_ = 0;
      isInplace_ = isInplace;
      isComplex_ = isComplex;
    }

    void setRun(int run) {
      run_ = run;
    }

    void addSizeDeviceDataBuffer(size_t s) {
      sizeDeviceDataBuffer_[run_] = s;
    }
    void addSizeDevicePlanBuffer(size_t s) {
      sizeDevicePlanBuffer_[run_] = s;
    }
    template<typename T_Index>
    void addValue(T_Index idx_val, double val) {
      int idx = static_cast<int>(idx_val);
      assert(idx<T_NumberValues);
      values_[run_][idx] = val;
    }

    size_t getDim() const { return dim_; }

    template<typename T_Index>
    auto getValue(T_Index idx_val) const {
      int idx = static_cast<int>(idx_val);
      assert(idx<T_NumberValues);
      return values_[run_][idx_val];
    }

    size_t getSizeDeviceDataBuffer() const {
      return sizeDeviceDataBuffer_[run_];
    }

    size_t getSizeDevicePlanBuffer() const {
      return sizeDevicePlanBuffer_[run_];
    }

    bool isInplace() const { return isInplace_; }
    bool isComplex() const { return isComplex_; }
    size_t getDimKind() const { return dimkind_; }
    auto getExtents() const { return extents_; }

  private:
    /// result object id
    //size_t id_ = 0;
    int run_ = 0;
    /// fft dimension
    size_t dim_ = 0;
    /// extents are 1=Arbitrary, 2=PowerOfTwo, 3=PowerOfOther
    size_t dimkind_ = 0;
    /// fft extents
    std::array<unsigned,3> extents_ = { {1} };
    /// size of device buffer allocated for data transfers
    BufferSizesT sizeDeviceDataBuffer_ = { {0} };
    /// size of device buffer allocated by FFT-Plan
    BufferSizesT sizeDevicePlanBuffer_ = { {0} };
    /// each run w values ( data[idx_run][idx_val] )
    ValuesT values_ = { {{ {0.0} }} };
    /// FFT Kind Inplace
    bool isInplace_ = false;
    /// FFT Kind Complex
    bool isComplex_ = false;
  };
}

#endif
