add_executable(Physarum
  Helpers.swift
  main.swift)
target_link_libraries(Physarum PRIVATE
  ModelSupport
  SwiftRT)


install(TARGETS Physarum
  DESTINATION bin)
