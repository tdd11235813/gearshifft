#include "core/benchmark.hpp"

// ----------------------------------------------------------------------------
template<typename... Types>
using List = gearshifft::List<Types...>;

#if defined(LiFFT_ENABLED)
#include "libraries/lifft/lifft.hpp"
using namespace gearshifft::LiFFT;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::false_type;

#elif defined(CUDA_ENABLED)
#include "libraries/cufft/cufft.hpp"
using namespace gearshifft::CuFFT;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::false_type;

#elif defined(OPENCL_ENABLED)
#include "libraries/clfft/clfft.hpp"
using namespace gearshifft::ClFFT;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::true_type;

#elif defined(FFTW_ENABLED)
#include "libraries/fftw/fftw.hpp"
using namespace gearshifft::fftw;
using FFTs              = List<Inplace_Real,
                               Inplace_Complex,
                               Outplace_Real,
                               Outplace_Complex>;
using Precisions        = List<float, double>;
using FFT_Is_Normalized = std::false_type;
#endif

// ----------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  try {
    gearshifft::Benchmark<Context> benchmark;

    benchmark.configure(argc, argv);
    benchmark.run<FFT_Is_Normalized, FFTs, Precisions>();

  }catch(const std::runtime_error& e){
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
