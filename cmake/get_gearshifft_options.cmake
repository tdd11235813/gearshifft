
macro(get_gearshifft_options result_var prefix)
  get_cmake_property(_variableNames VARIABLES)
  list (SORT _variableNames)
  foreach (_variableName ${_variableNames})
    if(_variableName MATCHES "GEARSHIFFT"
        AND NOT _variableName MATCHES "GEARSHIFFT_SUPERBUILD"
        AND NOT _variableName MATCHES "GEARSHIFFT_EXT")
      string(APPEND ${result_var} "${prefix}${_variableName}=${${_variableName}} ")
    endif()
  endforeach()
endmacro()
