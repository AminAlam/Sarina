cmake_minimum_required(VERSION 3.10)

project(Sarina)

add_library(Sarina SHARED ${CMAKE_CURRENT_SOURCE_DIR}/sarina/lib/cpp_backend.cpp)
install(TARGETS Sarina DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/sarina/cpp_backend/lib)
