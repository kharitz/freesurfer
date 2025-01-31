project(mris_spintest)

include_directories(${FS_INCLUDE_DIRS})

add_executable(mris_spintest mris_spintest.cpp)
target_link_libraries(mris_spintest utils)

install(TARGETS mris_spintest DESTINATION bin)
