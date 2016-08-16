#ifndef LIFFT_HPP_
#define LIFFT_HPP_

#include "libLiFFT/FFT_Kind.hpp"
#include "libLiFFT/FFT_Definition.hpp"
#include "libLiFFT/types/Real.hpp"
#include "libLiFFT/types/Complex.hpp"
#include "libLiFFT/mem/PlainPtrWrapper.hpp"


#ifdef CUDA_ENABLED
#include "cufft.hpp"
#endif
#ifdef OPENCL_ENABLED
#include "clfft.hpp"
#endif
#ifdef FFTW_ENABLED
#include "fftw.hpp"
#endif

namespace gearshifft {
namespace LiFFT {
  namespace traits{
  }

  #ifdef CUDA_ENABLED
  using gearshifft::CuFFT::Context;
  using FFT_Library = ::LiFFT::libraries::cuFFT::CuFFT<>;
  #endif
  #ifdef OPENCL_ENABLED
  using gearshifft::ClFFT::Context;
  #endif
  #ifdef FFTW_ENABLED
  using gearshifft::FFTW::Context;
  #endif


  /**
   * CuFFT plan and execution class.
   *
   * This class handles:
   * - {1D, 2D, 3D} x {R2C, C2R, C2C} x {inplace, outplace} x {float, double}.
   */
  template<typename T_FFT, // see fft_abstract.hpp (FFT_Inplace_Real, ...)
           typename T_Precision, // double, float
           size_t   NDim // 1..3
           >
  struct LiFFTImpl {
    using Extent = std::array<size_t,NDim>;

    static constexpr
     bool IsInplace = T_FFT::IsInplace;
    static constexpr
     bool IsComplex = T_FFT::IsComplex;
    static constexpr
     bool IsInplaceReal = IsInplace && IsComplex==false;
    using RealType = ::LiFFT::types::Real<T_Precision>;
    using ComplexType = ::LiFFT::types::Complex<T_Precision>;
    using RealOrComplexType  = typename std::conditional<IsComplex,
                                                         ComplexType,
                                                         RealType>::type;
    using FFTForward = typename std::conditional<IsComplex,
                                        ::LiFFT::FFT_Definition
                                          < ::LiFFT::FFT_Kind::Complex2Complex,
                                            NDim,
                                            T_Precision,
                                            std::true_type,
                                            IsInplace >,
                                        ::LiFFT::FFT_Definition
                                          < ::LiFFT::FFT_Kind::Real2Complex,
                                            NDim,
                                            T_Precision,
                                            std::true_type,
                                            IsInplace > >::type;
    using FFTBackward= typename std::conditional<IsComplex,
                                        ::LiFFT::FFT_Definition
                                          < ::LiFFT::FFT_Kind::Complex2Complex,
                                            NDim,
                                            T_Precision,
                                            std::false_type,
                                            IsInplace >,
                                        ::LiFFT::FFT_Definition
                                          < ::LiFFT::FFT_Kind::Complex2Real,
                                            NDim,
                                            T_Precision,
                                            std::false_type,
                                            IsInplace > >::type;

    size_t n_;        // =[1]*..*[dim]
    Extent extents_;
    Extent extents_complex_;
    RealOrComplexType* data_;
    ComplexType* data_complex_;

    size_t             data_size_;
    size_t             data_complex_size_;

    LiFFTImpl(const Extent& cextents) {
      extents_ = interpret_as::column_major(cextents);

      n_ = std::accumulate(extents_.begin(),
                           extents_.end(),
                           static_cast<size_t>(1),
                           std::multiplies<size_t>());
      size_t n_half = n_ / extents_[NDim-1] * (extents_[NDim-1]/2 + 1);
      data_size_ = ( IsInplaceReal ? 2*n_half : n_ ) * sizeof(RealOrComplexType);
      data_complex_size_ = ( IsInplace ? 0 : IsComplex ? n_ : n_half ) * sizeof(ComplexType);

      if(IsInplace==false){
        extents_complex_ = extents_;
        if(IsComplex==false)
          extents_complex_[NDim-1] = extents_complex_[NDim-1]/2+1;
      }
    }


    /**
     * Returns allocated memory on device for FFT
     */
    size_t getAllocSize() {
      return data_size_ + data_complex_size_;
    }

    /**
     * Returns data to be transfered to and from device for FFT
     */
    size_t getTransferSize() {
      return IsInplaceReal ? n_*sizeof(RealType) : data_size_;
    }

    /**
     * Returns estimated allocated memory on device for FFT plan
     */
    size_t getPlanSize() {
      size_t size1 = 0;
      size_t size2 = 0;
      return std::max(size1,size2);
    }

    // --- next methods are benchmarked ---

    /**
     * Allocate buffers on CUDA device
     */
    void malloc() {
    }

    // create FFT plan handle
    void init_forward() {
    }

    // recreates plan if needed
    void init_backward() {
    }

    // data_ and data_complex_ should be device pointer to have better control
    void execute_forward() {
      auto inWrapped = FFTForward::wrapInput( LiFFT::mem::wrapPtr<IsComplex>(
                                                data_,
                                                extents_) );
      if(IsInplace) {
        auto fft = ::LiFFT::makeFFT<FFT_Library>(inWrapped);
        fft(inWrapped);
      }else{
        auto outWrapped = FFTForward::wrapOutput(LiFFT::mem::wrapPtr<true,true>(
                                                   data_complex_,
                                                   extents_complex_));

        auto fft = ::LiFFT::makeFFT<FFT_Library>(inWrapped, outWrapped);
        fft(inWrapped, outWrapped);
      }
    }

    void execute_backward() {
    }

    template<typename THostData>
    void upload(THostData* input) {
//      std::copy();
      data_ = reinterpret_cast<RealType*>(input);
    }

    template<typename THostData>
    void download(THostData* output) {
      output = reinterpret_cast<THostData*>(data_);
    }

    void destroy() {
    }

  };

  typedef gearshifft::FFT<gearshifft::FFT_Inplace_Real, LiFFTImpl, TimerGPU> Inplace_Real;
  typedef gearshifft::FFT<gearshifft::FFT_Outplace_Real, LiFFTImpl, TimerGPU> Outplace_Real;
  typedef gearshifft::FFT<gearshifft::FFT_Inplace_Complex, LiFFTImpl, TimerGPU> Inplace_Complex;
  typedef gearshifft::FFT<gearshifft::FFT_Outplace_Complex, LiFFTImpl, TimerGPU> Outplace_Complex;

}
}

#endif
