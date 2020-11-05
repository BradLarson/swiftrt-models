add_executable(Fractals
  ComplexRange.swift
  ImageUtilities.swift
  JuliaSet.swift
  MandelbrotSet.swift
  main.swift)
target_link_libraries(Fractals PRIVATE
  ArgumentParser
  ModelSupport
  SwiftRT)

if(SWIFTRT_ENABLE_CUDA)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(Fractals PRIVATE
  CUDA::cudart)
endif()

install(TARGETS Fractals
  DESTINATION bin)
