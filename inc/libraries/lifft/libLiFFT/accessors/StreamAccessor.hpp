/* This file is part of libLiFFT.
 *
 * libLiFFT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libLiFFT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libLiFFT.  If not, see <www.gnu.org/licenses/>.
 */
 
#pragma once

#include <cstring>

namespace LiFFT {
namespace accessors {

    /**
     * Provides delimiters for (up to) 2D text-files
     */
    struct TextDelimiters
    {
        static constexpr unsigned numDims = 2;

        const char*
        operator[](unsigned dim) const
        {
            if(dim == 0)
                return "\n";
            else
                return " ";
        }
    };

    /**
     * Trait to provide the number of chars a delimiter (or any other object instance) takes
     * Provides a static get(T) function that returns the size in chars
     * Used to skip over delimiters in a stream
     */
    template< typename T >
    struct GetNumChars;

    template<>
    struct GetNumChars< const char* >
    {
        static size_t
        get(const char* s){
            return strlen(s);
        }
    };

    /**
     * Simple stream accessor that writes the value into the stream via <<-operator
     * and reads via the >>-operator into a variable of the given ReturnType
     *
     * \tparam T_ReturnType type returned by the read-operator [float]
     * \tparam T_Delimiters Must provide a []-operator(dimension) method to return a delimiter separating values from that dimension
     *
     */
    template< typename T_ReturnType = float, class T_Delimiters = TextDelimiters >
    struct StreamAccessor
    {
        const T_Delimiters m_delimiters = T_Delimiters();

        template< class T_Index, class T_Stream, typename T_Value >
        void operator()(const T_Index& /*idx*/, T_Stream& stream, T_Value&& value)
        {
            stream << std::forward<T_Value>(value);
        }

        template< class T_Index, class T_Stream >
        T_ReturnType operator()(const T_Index& /*idx*/, T_Stream& stream)
        {
            T_ReturnType result;
            stream >> result;
            return result;
        }

        const T_Delimiters&
        getDelimiters()
        {
            return m_delimiters;
        }

        template< class T_Stream >
        void
        skipDelimiter(T_Stream& stream, unsigned idx)
        {
            size_t numChars = GetNumChars< decltype(m_delimiters[idx])>::get(m_delimiters[idx]);
            stream.ignore(numChars);
        }
    };

    template< typename T_ReturnType = float >
    using StringStreamAccessor = StreamAccessor< T_ReturnType >;

}  // namespace accessors
}  // namespace LiFFT
