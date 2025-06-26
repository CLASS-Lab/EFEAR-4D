cmake_minimum_required(VERSION 3.0.2)
project(efear)

set(CMAKE_CXX_FLAGS "-std=c++17 -O2 ${SSE_FLAGS} -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")

# pcl 1.7 causes a segfault when it is built with debug mode
set(CMAKE_BUILD_TYPE "RELEASE")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  image_transport

  pcl_ros
  pcl_conversions

  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs

  message_generation
)

# find_package(GTSAM REQUIRED QUIET)
find_package(PCL 1.10 REQUIRED)
find_package(OpenCV REQUIRED QUIET)
# 合并 Boost 查找
find_package(Boost REQUIRED COMPONENTS 
  system filesystem date_time timer thread serialization chrono
)
find_package(Ceres REQUIRED)


include_directories(SYSTEM ${G2O_INCLUDE_DIR} ${G2O_INCLUDE_DIRS})
link_directories(${G2O_LIBRARY_DIRS})

add_message_files(
  FILES
  cloud_msgs.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  save_map.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS   std_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
  DEPENDS PCL
)



include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
	# ${GTSAM_INCLUDE_DIR}
)


link_directories(
	include
	${OpenCV_LIBRARY_DIRS}
	${PCL_LIBRARY_DIRS}
	# ${GTSAM_LIBRARY_DIRS}
)

add_executable(PreProcessing src/PreProcessing.cpp)
add_dependencies(PreProcessing  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(PreProcessing 
${catkin_LIBRARIES} 
${PCL_LIBRARIES} 
${OpenCV_LIBRARIES}
${CERES_LIBRARIES}
)

add_executable(ScanMatching src/ScanMatching.cpp
  src/efear/voxelgrid.cpp
  src/efear/pointnormal.cpp
  src/efear/registration.cpp
  src/efear/n_scan_normal.cpp
  src/efear/utils.cpp 
  )
add_dependencies(ScanMatching  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ScanMatching 
${catkin_LIBRARIES} 
${PCL_LIBRARIES} 
${OpenCV_LIBRARIES}
${CERES_LIBRARIES}
)
