cmake_minimum_required(VERSION 3.0.2)
project(detect)

find_package(catkin REQUIRED COMPONENTS dynamic_reconfigure)

catkin_python_setup()

generate_dynamic_reconfigure_options(
	cfg/DetectLane.cfg
	cfg/DetectTrafficLight.cfg
	cfg/DetectLevel.cfg
)

catkin_package(CATKIN_DEPENDS dynamic_reconfigure)

catkin_install_python(PROGRAMS
  	nodes/detect_lane
	nodes/detect_traffic_light
	nodes/detect_intersection
	nodes/detect_construction
	nodes/detect_parking
	nodes/detect_level
	nodes/detect_tunnel
 	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY yaml cfg data launch image
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
