include(FetchContent)
FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.4)
FetchContent_MakeAvailable(Corrosion)
corrosion_import_crate(MANIFEST_PATH Cargo.toml CRATES qiskit-c-ext)
enable_testing()
add_executable(test_main test/c/test_main.c)
target_link_libraries(test_main qiskit_c_ext)
add_test(NAME test_main COMMAND test_main)
