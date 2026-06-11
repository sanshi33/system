cmake_minimum_required(VERSION 3.21)

if(WIN32)
    set(_stitch_default_vcpkg_root "G:/vcpkg-2025.10.17")
    if(NOT EXISTS "${_stitch_default_vcpkg_root}" AND EXISTS "D:/VSCode/vcpkg-2025.10.17")
        set(_stitch_default_vcpkg_root "D:/VSCode/vcpkg-2025.10.17")
    endif()
    set(STITCH_VCPKG_ROOT "${_stitch_default_vcpkg_root}" CACHE PATH "Path to the vcpkg root used by this project")
    set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg triplet for dependency resolution")

    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        set(_stitch_vcpkg_toolchain "${STITCH_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        if(EXISTS "${_stitch_vcpkg_toolchain}")
            set(CMAKE_TOOLCHAIN_FILE "${_stitch_vcpkg_toolchain}" CACHE FILEPATH "vcpkg toolchain file")
        else()
            message(WARNING "vcpkg toolchain was not found at ${_stitch_vcpkg_toolchain}. Configure CMAKE_TOOLCHAIN_FILE manually if needed.")
        endif()
    endif()

    set(STITCH_VCPKG_INSTALLED_DIR "${STITCH_VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}" CACHE PATH "Resolved vcpkg installed directory")
    if(EXISTS "${STITCH_VCPKG_INSTALLED_DIR}")
        set(_stitch_opencv_config_dir "${STITCH_VCPKG_INSTALLED_DIR}/share/opencv4")
        set(_stitch_eigen3_config_dir "${STITCH_VCPKG_INSTALLED_DIR}/share/eigen3")

        list(PREPEND CMAKE_PREFIX_PATH
            "${STITCH_VCPKG_INSTALLED_DIR}"
            "${_stitch_opencv_config_dir}"
            "${_stitch_eigen3_config_dir}"
        )

        # 对 VS Code / CMake Tools 更友好：即使没有显式选中 preset，
        # 也尽量把 OpenCV / Eigen3 的 package config 目录指清楚。
        if(NOT DEFINED OpenCV_DIR AND EXISTS "${_stitch_opencv_config_dir}/OpenCVConfig.cmake")
            set(OpenCV_DIR "${_stitch_opencv_config_dir}" CACHE PATH "OpenCV package config directory")
        endif()

        if(NOT DEFINED Eigen3_DIR AND EXISTS "${_stitch_eigen3_config_dir}/Eigen3Config.cmake")
            set(Eigen3_DIR "${_stitch_eigen3_config_dir}" CACHE PATH "Eigen3 package config directory")
        endif()
    endif()
endif()

project(stitch_refactor VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" PINJIE_PROJECT_ROOT_CMAKE)
add_compile_definitions(PINJIE_PROJECT_ROOT="${PINJIE_PROJECT_ROOT_CMAKE}")

option(PINJIE_BUILD_GUI "Build the Qt Widgets GUI shell." ON)
option(PINJIE_ENABLE_GALAXY_CAMERA "Build Galaxy SDK camera acquisition support." ON)
set(_pinjie_default_galaxy_sdk_root "G:/GalaxySDK")
if(NOT EXISTS "${_pinjie_default_galaxy_sdk_root}" AND EXISTS "E:/GalaxySDK")
    set(_pinjie_default_galaxy_sdk_root "E:/GalaxySDK")
endif()
set(GALAXY_SDK_ROOT "${_pinjie_default_galaxy_sdk_root}" CACHE PATH "Galaxy SDK installation root")
set(GALAXY_SDK_DEVELOPMENT "${GALAXY_SDK_ROOT}/Development" CACHE PATH "Galaxy SDK Development directory")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_BINARY_DIR}/bin/${cfg}")
endforeach()

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs calib3d features2d)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)

set(PINJIE_GALAXY_CAMERA_TARGET "")
if(PINJIE_ENABLE_GALAXY_CAMERA)
    set(_galaxy_cpp_include "${GALAXY_SDK_DEVELOPMENT}/C++ SDK/inc")
    set(_galaxy_cpp_lib_dir "${GALAXY_SDK_DEVELOPMENT}/C++ SDK/lib/x64")
    set(_galaxy_cpp_lib "${_galaxy_cpp_lib_dir}/GxIAPICPPEx.lib")
    set(_galaxy_api_dll_dir "${GALAXY_SDK_ROOT}/APIDll/Win64")

    if(EXISTS "${_galaxy_cpp_include}" AND EXISTS "${_galaxy_cpp_lib}")
        add_library(pinjie_acquisition STATIC
            acquisition/CameraCaptureService.cpp
            acquisition/GalaxyCameraDevice.cpp
        )
        target_compile_features(pinjie_acquisition PUBLIC cxx_std_17)
        target_include_directories(pinjie_acquisition
            PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}"
            SYSTEM PRIVATE
                "${_galaxy_cpp_include}"
        )
        target_link_libraries(pinjie_acquisition
            PUBLIC
                opencv_core
                opencv_imgcodecs
                opencv_imgproc
            PRIVATE
                "${_galaxy_cpp_lib}"
        )
        target_link_directories(pinjie_acquisition PRIVATE "${_galaxy_cpp_lib_dir}")
        target_compile_definitions(pinjie_acquisition PUBLIC PINJIE_ENABLE_GALAXY_CAMERA=1)
        file(TO_CMAKE_PATH "${GALAXY_SDK_ROOT}" PINJIE_GALAXY_SDK_ROOT_CMAKE)
        target_compile_definitions(pinjie_acquisition
            PRIVATE
                PINJIE_GALAXY_SDK_ROOT="${PINJIE_GALAXY_SDK_ROOT_CMAKE}"
        )

        if(MSVC)
            target_compile_options(pinjie_acquisition PRIVATE /utf-8 /permissive- /W4 /wd4828)
            target_compile_definitions(pinjie_acquisition PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
            target_link_options(pinjie_acquisition INTERFACE /DELAYLOAD:GxIAPICPPEx.dll)
            target_link_libraries(pinjie_acquisition INTERFACE delayimp)
        else()
            target_compile_options(pinjie_acquisition PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        add_executable(pinjie_camera_probe acquisition/camera_probe.cpp)
        target_compile_features(pinjie_camera_probe PRIVATE cxx_std_17)
        target_link_libraries(pinjie_camera_probe PRIVATE pinjie_acquisition)

        if(MSVC)
            target_compile_options(pinjie_camera_probe PRIVATE /utf-8 /permissive- /W4)
            target_compile_definitions(pinjie_camera_probe PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
        else()
            target_compile_options(pinjie_camera_probe PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        if(WIN32 AND EXISTS "${_galaxy_api_dll_dir}")
            set(_pinjie_galaxy_runtime_dlls
                "${_galaxy_api_dll_dir}/GxIAPICPPEx.dll"
                "${_galaxy_api_dll_dir}/GxIAPICPP.dll"
                "${_galaxy_api_dll_dir}/GxIAPI.dll"
                "${_galaxy_api_dll_dir}/DxImageProc.dll"
                "${_galaxy_api_dll_dir}/DxMediaProc.dll"
            )
            foreach(_pinjie_galaxy_dll IN LISTS _pinjie_galaxy_runtime_dlls)
                if(EXISTS "${_pinjie_galaxy_dll}")
                    add_custom_command(TARGET pinjie_camera_probe POST_BUILD
                        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                            "${_pinjie_galaxy_dll}"
                            "$<TARGET_FILE_DIR:pinjie_camera_probe>"
                        VERBATIM
                    )
                endif()
            endforeach()
        endif()

        set(PINJIE_GALAXY_CAMERA_TARGET pinjie_acquisition)
    else()
        message(WARNING "Galaxy SDK C++ headers or library were not found. Camera acquisition target will not be built.")
    endif()
endif()

set(STITCH_CORE_SOURCES
    cad_design/DesignProfileAlignment.cpp
    cad_design/DesignProfileFunction.cpp
    core/SubpixelEdgeDetector.cpp
    stitch/Alignment.cpp
    stitch/Blending.cpp
    stitch/DebugVis.cpp
    stitch/EdgeProcessing.cpp
    stitch/GeometryUtils.cpp
    stitch/IO.cpp
    stitch/Pipeline.cpp
    stitch/PoseGraphOptimizer.cpp
    stitch/StitchService.cpp
)

add_library(stitch_core STATIC ${STITCH_CORE_SOURCES})
target_compile_features(stitch_core PUBLIC cxx_std_17)
target_include_directories(stitch_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(stitch_core PRIVATE /utf-8 /permissive- /W4)
    target_compile_definitions(stitch_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(stitch_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_link_libraries(stitch_core
    PUBLIC
        Eigen3::Eigen
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
)

add_executable(stitch_app main_clean.cpp)
target_compile_features(stitch_app PRIVATE cxx_std_17)
target_link_libraries(stitch_app PRIVATE stitch_core)

if(MSVC)
    target_compile_options(stitch_app PRIVATE /utf-8 /permissive- /W4)
    target_compile_definitions(stitch_app PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(stitch_app PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
    add_custom_command(TARGET stitch_app POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            $<TARGET_RUNTIME_DLLS:stitch_app>
            $<TARGET_FILE_DIR:stitch_app>
        COMMAND_EXPAND_LISTS
    )
endif()

add_library(pinjie_app STATIC
    app/StitchWorkflowService.cpp
    calibration/CalibrationService.cpp
    calibration/CircleEdgeResponse.cpp
    calibration/TelecentricCalibIO.cpp
    calibration/TelecentricCalibrator.cpp
    calibration/TelecentricUndistort.cpp
    report/QualitySummaryBuilder.cpp
)
target_compile_features(pinjie_app PUBLIC cxx_std_17)
target_include_directories(pinjie_app
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(pinjie_app
    PUBLIC
        stitch_core
        Ceres::ceres
        opencv_calib3d
        opencv_features2d
)

if(MSVC)
    target_compile_options(pinjie_app PRIVATE /utf-8 /permissive- /W4)
    target_compile_definitions(pinjie_app PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(pinjie_app PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(pinjie_cli cli/main.cpp)
target_compile_features(pinjie_cli PRIVATE cxx_std_17)
target_link_libraries(pinjie_cli PRIVATE pinjie_app)

if(MSVC)
    target_compile_options(pinjie_cli PRIVATE /utf-8 /permissive- /W4)
    target_compile_definitions(pinjie_cli PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(pinjie_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
    add_custom_command(TARGET pinjie_cli POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            $<TARGET_RUNTIME_DLLS:pinjie_cli>
            $<TARGET_FILE_DIR:pinjie_cli>
        COMMAND_EXPAND_LISTS
    )
endif()

add_executable(standard_sphere_loop_eval
    standard_sphere_loop/main.cpp
    standard_sphere_loop/StandardSphereLoopExperiment.cpp
    standard_sphere_loop/StandardSphereLoopExperiment.h
)
target_compile_features(standard_sphere_loop_eval PRIVATE cxx_std_17)
target_link_libraries(standard_sphere_loop_eval PRIVATE stitch_core)

if(MSVC)
    target_compile_options(standard_sphere_loop_eval PRIVATE /utf-8 /permissive- /W4)
    target_compile_definitions(standard_sphere_loop_eval PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(standard_sphere_loop_eval PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
    add_custom_command(TARGET standard_sphere_loop_eval POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            $<TARGET_RUNTIME_DLLS:standard_sphere_loop_eval>
            $<TARGET_FILE_DIR:standard_sphere_loop_eval>
        COMMAND_EXPAND_LISTS
    )
endif()

if(PINJIE_BUILD_GUI)
    set(CMAKE_AUTOMOC ON)
    find_package(Qt6 CONFIG REQUIRED COMPONENTS Core Gui Widgets)

    qt_add_executable(pinjie_gui
        MANUAL_FINALIZATION
        gui/main.cpp
        gui/CalibrationConfigPanel.h
        gui/CalibrationConfigPanel.cpp
        gui/CalibrationWorker.h
        gui/CalibrationWorker.cpp
        gui/CameraCaptureWorker.h
        gui/CameraCaptureWorker.cpp
        gui/ImageViewer.h
        gui/ImageViewer.cpp
        gui/StepTableModel.h
        gui/StepTableModel.cpp
        gui/StepDetailPanel.h
        gui/StepDetailPanel.cpp
        gui/RunConfigPanel.h
        gui/RunConfigPanel.cpp
        gui/StitchWorker.h
        gui/StitchWorker.cpp
        gui/MainWindow.h
        gui/MainWindow.cpp
        gui/QtImageUtils.h
    )
    target_compile_features(pinjie_gui PRIVATE cxx_std_17)
    target_link_libraries(pinjie_gui
        PRIVATE
            pinjie_app
            Qt6::Core
            Qt6::Gui
            Qt6::Widgets
    )
    add_dependencies(pinjie_gui standard_sphere_loop_eval)
    if(PINJIE_GALAXY_CAMERA_TARGET)
        target_link_libraries(pinjie_gui PRIVATE ${PINJIE_GALAXY_CAMERA_TARGET})
    endif()

    if(MSVC)
        target_compile_options(pinjie_gui PRIVATE /utf-8 /permissive- /W4)
        target_compile_definitions(pinjie_gui PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    else()
        target_compile_options(pinjie_gui PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    qt_finalize_executable(pinjie_gui)

    if(WIN32)
        add_custom_command(TARGET pinjie_gui POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                $<TARGET_RUNTIME_DLLS:pinjie_gui>
                $<TARGET_FILE_DIR:pinjie_gui>
            COMMAND_EXPAND_LISTS
        )

        if(PINJIE_GALAXY_CAMERA_TARGET AND EXISTS "${_galaxy_api_dll_dir}")
            foreach(_pinjie_galaxy_dll IN LISTS _pinjie_galaxy_runtime_dlls)
                if(EXISTS "${_pinjie_galaxy_dll}")
                    add_custom_command(TARGET pinjie_gui POST_BUILD
                        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                            "${_pinjie_galaxy_dll}"
                            "$<TARGET_FILE_DIR:pinjie_gui>"
                        VERBATIM
                    )
                endif()
            endforeach()
        endif()

        set(_pinjie_qt_release_plugin_root "${STITCH_VCPKG_INSTALLED_DIR}/Qt6/plugins")
        set(_pinjie_qt_debug_plugin_root "${STITCH_VCPKG_INSTALLED_DIR}/debug/Qt6/plugins")
        if(EXISTS "${_pinjie_qt_release_plugin_root}" OR EXISTS "${_pinjie_qt_debug_plugin_root}")
            set(_pinjie_qwindows_plugin
                "$<$<CONFIG:Debug>:${_pinjie_qt_debug_plugin_root}/platforms/qwindowsd.dll>$<$<NOT:$<CONFIG:Debug>>:${_pinjie_qt_release_plugin_root}/platforms/qwindows.dll>"
            )
            set(_pinjie_qgif_plugin
                "$<$<CONFIG:Debug>:${_pinjie_qt_debug_plugin_root}/imageformats/qgifd.dll>$<$<NOT:$<CONFIG:Debug>>:${_pinjie_qt_release_plugin_root}/imageformats/qgif.dll>"
            )
            set(_pinjie_qico_plugin
                "$<$<CONFIG:Debug>:${_pinjie_qt_debug_plugin_root}/imageformats/qicod.dll>$<$<NOT:$<CONFIG:Debug>>:${_pinjie_qt_release_plugin_root}/imageformats/qico.dll>"
            )
            set(_pinjie_qjpeg_plugin
                "$<$<CONFIG:Debug>:${_pinjie_qt_debug_plugin_root}/imageformats/qjpegd.dll>$<$<NOT:$<CONFIG:Debug>>:${_pinjie_qt_release_plugin_root}/imageformats/qjpeg.dll>"
            )

            add_custom_command(TARGET pinjie_gui POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:pinjie_gui>/platforms"
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${_pinjie_qwindows_plugin}"
                    "$<TARGET_FILE_DIR:pinjie_gui>/platforms"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:pinjie_gui>/imageformats"
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${_pinjie_qgif_plugin}"
                    "${_pinjie_qico_plugin}"
                    "${_pinjie_qjpeg_plugin}"
                    "$<TARGET_FILE_DIR:pinjie_gui>/imageformats"
                COMMENT "Deploying Qt plugins for pinjie_gui"
                VERBATIM
            )
        else()
            message(WARNING "Qt plugin directories were not found. pinjie_gui may fail to start if Qt platform plugins are missing.")
        endif()
    endif()
endif()
