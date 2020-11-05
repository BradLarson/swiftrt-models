add_executable(Physarum
  Helpers.swift
  main.swift)
target_link_libraries(Physarum PRIVATE
  ModelSupport
  SwiftRT)

if(SWIFTRT_ENABLE_CUDA)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(Physarum PRIVATE
  CUDA::cudart)
endif()

install(TARGETS Physarum
  DESTINATION bin)
