add_executable(Physarum
  main.swift)
target_link_libraries(Physarum PRIVATE
  ModelSupport
  SwiftRT)


install(TARGETS Physarum
  DESTINATION bin)
