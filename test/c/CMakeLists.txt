file(GLOB TestsToRun LIST_DIRECTORIES FALSE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test_*.c)
create_test_sourcelist(Tests AllTests.c ${TestsToRun})
add_executable(AllTests ${Tests})

target_link_libraries(AllTests ${Qiskit})

foreach(test ${TestsToRun})
    get_filename_component(TName ${test} NAME_WE)
    add_test(NAME ${TName} COMMAND AllTests ${TName})
endforeach()
