cmake_minimum_required(VERSION 3.18)
project(PALLAS_VESTA LANGUAGES CXX CUDA)

# Use C++17 and CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architectures if needed
# set(CMAKE_CUDA_ARCHITECTURES 75)  # For RTX 3060 (Ampere)

# Add main source
add_executable(main_exec src/main.cu)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Glob all *_test.cu files in tests/
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*_test.cu)

# Add test executable
add_executable(cuda_tests ${TEST_SOURCES})

# Link GTest and CUDA
target_link_libraries(cuda_tests gtest_main)
target_compile_features(cuda_tests PUBLIC cxx_std_17)
set_target_properties(cuda_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add include directories if needed
target_include_directories(cuda_tests PRIVATE src)

# Test discovery
include(GoogleTest)
gtest_discover_tests(cuda_tests)
configure_file(tests/constant_sage.py constant_sage.py COPYONLY)