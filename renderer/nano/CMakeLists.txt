embed_ptx(
	OUTPUT_TARGET
		nano-ptx
	PTX_LINK_LIBRARIES
		owl::owl
		B3D::Renderer::Library
	SOURCES
		NanoRenderer.cu
		SampleAccumulators.h
		SamplerMapper.h
	#DEVICE_DEBUG
)


set_property(TARGET nano-ptx PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET nano-ptx_ptx PROPERTY FOLDER "Renderer/PTX Targets")

embed_optixir(
	OUTPUT_TARGET
		nano-optixir
	OPTIXIR_LINK_LIBRARIES
		owl::owl	
		B3D::Renderer::Library
	SOURCES
		NanoRenderer.cu
		SampleAccumulators.h
		SamplerMapper.h
	#DEVICE_DEBUG
)


set_property(TARGET nano-optixir PROPERTY FOLDER "Renderer/PTX Targets")
set_property(TARGET nano-optixir_optixir PROPERTY FOLDER "Renderer/PTX Targets")

set_source_files_properties(
	NanoRenderer.cu
	PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
 )

 

find_package(OpenVDB CONFIG REQUIRED COMPONENTS nanovdb)
find_package(imgui CONFIG REQUIRED)

# set_source_files_properties(
#	 FoveatedRendering.cu
#	 PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
# )


add_library(cudaCode FoveatedRendering.cu FoveatedRendering.h FoveatedHelper.cuh FoveatedHelper.cu)
target_compile_features(cudaCode PUBLIC cxx_std_20)
target_include_directories(cudaCode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cudaCode
	PRIVATE
		B3D::Renderer::Library
)
set_property(TARGET cudaCode PROPERTY CUDA_STANDARD 20)


add_b3d_renderer(
	NAME
		Nano
	SOURCES
		NanoRenderer.cpp
	HEADERS
		NanoRenderer.h
	LIBRARIES
		cudaCode
		nano-ptx
		nano-optixir
		OpenVDB::nanovdb
		imgui::imgui
		NanoCutterParser
		B3D::Tools::ColorMap
)
