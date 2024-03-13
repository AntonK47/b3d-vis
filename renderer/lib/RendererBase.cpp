#include "RendererBase.h"
#include <vector>

#include <ranges>

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"

using namespace b3d::renderer;

std::vector<RendererRegistryEntry> b3d::renderer::registry;

auto RendererBase::initialize(RenderingDataBuffer* renderData, const DebugInitializationInfo& debugInitializationInfo)
	-> void
{
	renderData_ = renderData;
	debugInfo_ = debugInitializationInfo;

	for (const auto& feature : renderFeatures_)
	{
		feature->initialize(*renderData);
	}

	onInitialize();
}

auto RendererBase::deinitialize() -> void
{
	onDeinitialize();
	for (const auto& feature : renderFeatures_)
	{
		feature->deinitialize();
	}
}

auto RendererBase::gui() -> void
{
	onGui();

	ImGui::Begin("Features");

	for (const auto& feature : renderFeatures_)
	{
		if (feature->hasGui())
		{
			if (ImGui::CollapsingHeader(feature->featureName().c_str()))
			{
				feature->gui();
			}
		}
	}

	ImGui::End();
}

auto RendererBase::render() -> void
{
	for (const auto& feature : renderFeatures_)
	{
		feature->beginUpdate();
	}

	onRender();

	for (const auto& feature : renderFeatures_ | std::views::reverse)
	{
		feature->endUpdate();
	}
}
