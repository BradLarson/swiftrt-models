add_subdirectory(STBImage)

add_library(ModelSupport
  AnimatedImage.swift
  FileManagement.swift
  FileSystem.swift
  FoundationFileSystem.swift
  GIF.swift
  Image.swift
  Stderr.swift)
set_target_properties(ModelSupport PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_options(ModelSupport PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
target_link_libraries(ModelSupport PUBLIC
  Numerics
  STBImage
  SwiftRT)


install(TARGETS ModelSupport
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES
  $<TARGET_PROPERTY:ModelSupport,Swift_MODULE_DIRECTORY>/ModelSupport.swiftdoc
  $<TARGET_PROPERTY:ModelSupport,Swift_MODULE_DIRECTORY>/ModelSupport.swiftmodule
  DESTINATION lib/swift$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:_static>/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/${swift_arch})
