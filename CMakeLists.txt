enable_testing()
find_library(Qiskit qiskit_c_ext PATHS target/release REQUIRED)
add_subdirectory(test/c)
