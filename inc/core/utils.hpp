#ifndef UTILS_HPP
#define UTILS_HPP

#include <functional>

#ifndef GEARSHIFFT_STRINGIFY
#define GEARSHIFFT_STRINGIFY_EXPAND(tok) #tok
#define GEARSHIFFT_STRINGIFY(s) GEARSHIFFT_STRINGIFY_EXPAND(s)
#endif

namespace gearshifft {
  namespace utils {

    inline
    bool powerOf(size_t e, double b) {
      if(e==0)
        return false;
      double a = static_cast<double>(e);
      double p = std::floor(std::log(a)/std::log(b)+0.5);
      return std::fabs(std::pow(b,p)-a)<0.0001;
    }

    inline
    bool isNotOnlyDivBy_2_3_5(size_t e) {
      if(e==2 || e==3 || e==5)
        return false;
      size_t t = e;
      size_t sqr = static_cast<size_t>(std::sqrt(e));
      for( size_t d=2; d<=sqr; ++d ) {
        // if e contains divisors > 5
        if( e%d == 0 && d>5 )
          return true;
        // e = e / d^q
        while( e%d == 0 ) {
          e /= d;
        }
      }
      return e==t || e>5; // e might be a prime or a divisor>5
    }

    inline
    bool isNotOnlyDivBy_2_3_5_7(size_t e) {
      if(e==2 || e==3 || e==5 || e==7)
        return false;
      size_t t = e;
      size_t sqr = static_cast<size_t>(std::sqrt(e));
      for( size_t d=2; d<=sqr; ++d ) {
        // if e contains divisors > 7
        if( e%d == 0 && d>7 )
          return true;
        // e = e / d^q
        while( e%d == 0 ) {
          e /= d;
        }
      }
      return e==t || e>7; // e might be a prime or a divisor>7
    }

    /**
     * @retval 1 arbitrary extents
     * @retval 2 power-of-two extents
     * @retval 3 extents are combination of powers of (3,5,7) {3^r * 5^s * 7^t}
     */
    template<size_t TDim, typename TExtents>
    inline
    size_t computeDimkind(const TExtents& extents) {
      bool p2=true;
      for( size_t k=0; k<TDim; ++k ) {
        p2 &= powerOf(extents[k], 2.0);
      }
      if(p2)
        return 2;
      for( size_t k=0; k<TDim; ++k ) {
        if( isNotOnlyDivBy_2_3_5_7(extents[k]) ){
          return 1;
        }
      }
      return 3;
    }
    template<typename TExtents, typename TIdx = typename TExtents::value_type>
    constexpr
    TIdx extentsProduct(const TExtents& extents) {
      return std::accumulate(extents.begin(),
                             extents.end(),
                             1,
                             std::multiplies<TIdx>());
    }

    template<typename T, typename std::enable_if<std::is_floating_point<T>::value,int>::type=0>
    inline
    T multiply(T val, double fac) {
      return val*fac;
    }

    template<typename T, typename std::enable_if<std::is_floating_point<T>::value==false,int>::type=0>
    inline
    T multiply(T val,
               double fac) {
      using TVal = typename T::value_type;
      return
        std::complex<TVal>{val.real()*static_cast<TVal>(fac), val.imag()*static_cast<TVal>(fac)};
    }

    /**
     * Prints data as vector/matrix. For debugging purposes.
     * @param out output stream object
     * @param data data with real or complex data
     * @param extents extents of data matrix, read in xyz order
     * @param offset output with symmetric offset per dim
     * @param count number of elements to be printed per dim
     */
    template<typename TStream, typename T, typename TExtents>
    inline
    void print(TStream& out,
               T* data,
               TExtents extents,
               typename TExtents::value_type offset,
               typename TExtents::value_type count,
               double factor = 1.0) {
      using TIdx = typename TExtents::value_type;
      static constexpr TIdx NDim = extents.size();

      if(NDim==1) {
        TIdx n = extents[0]<offset+count ? extents[0] : offset+count;
        out << "0: ";
        for(TIdx i=offset; i<n; ++i) {
          out << multiply(data[i],factor) << " ";
        }
        out << "\n";
      } else {
        TIdx nx = extents[0]<offset+count ? extents[0] : offset+count;
        TIdx ny = extents[1]<offset+count ? extents[1] : offset+count;
        for(TIdx i=offset; i<ny; ++i) {
          out << i << ": ";
          for(TIdx j=offset; j<nx; ++j) {
            out << multiply(data[i*extents[0]+j],factor) << " ";
          }
          out << "\n";
        }
      }
    }

  } // utils
} // gearshifft

#endif /* UTILS_HPP */
