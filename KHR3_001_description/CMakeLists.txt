cmake_minimum_required(VERSION 3.8)
project(KHR3_001_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  meshes
  urdf
  vscode_preview
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
