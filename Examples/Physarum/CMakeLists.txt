add_executable(Physarum
  Helpers.swift
  main.swift)
target_link_libraries(Physarum PRIVATE
  ModelSupport
  SwiftRT)

if(SWIFTRT_ENABLE_CUDA)
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
      _CUDAToolkit_find_and_add_import_lib(cudnn)
endif()
target_link_libraries(Physarum PRIVATE
  CUDA::cudart
  CUDA::cudnn)
endif()

install(TARGETS Physarum
  DESTINATION bin)
