cmake_minimum_required(VERSION 2.6)

project(proyrcto_vision3d)

# Octomap
# Configure the path to the octomap library
# /home/miguelmg/octomap/octomap-devel/lib
# /home/miguelmg/octomap/octomap-devel/octomap

#/usr/lib
#/usr/include/octomap

include_directories(/home/miguelmg/octomap/octomap-devel/octomap/include)
link_directories(/home/miguelmg/octomap/octomap-devel/lib)
set(LIBS_OCTOMAP octomap octomath)

# Partial model
include_directories(/home/miguelmg/vpl/partialmodel)
link_directories(/home/miguelmg/vpl/lib)
set(LIB_PARTIAL_MODEL PartialModel)


add_executable(vpl_test vpl_test.cpp)
target_link_libraries(vpl_test  ${LIB_PARTIAL_MODEL} ${LIBS_OCTOMAP})


