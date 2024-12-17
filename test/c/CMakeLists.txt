add_executable(test_main test_main.c)
# FIXME: actually use find_library to link against qiskit_c_ext
target_link_libraries(test_main ${CMAKE_SOURCE_DIR}/target/release/libqiskit_c_ext.so)
add_test(NAME test_main COMMAND test_main)
