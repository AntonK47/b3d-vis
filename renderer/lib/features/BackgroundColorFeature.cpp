#include "BackgroundColorFeature.h"

#include <format>

#include "imgui.h"

b3d::renderer::BackgroundColorFeature::BackgroundColorFeature(const std::string& name) : RenderFeature{ name }
{
}

b3d::renderer::BackgroundColorFeature::BackgroundColorFeature(const std::string& name, std::array<ColorRGB, 2> colors)
	: RenderFeature{ name }, defaultColors_{ ColorRGBA{ colors[0].r, colors[0].g, colors[0].b, 1.0f },
											 ColorRGBA{ colors[1].r, colors[1].g, colors[1].b, 1.0f } }
{
}

auto b3d::renderer::BackgroundColorFeature::beginUpdate() -> void
{
	coloringInfo_ = sharedParameters_->get<ColoringInfo>("coloringInfo");
}

auto b3d::renderer::BackgroundColorFeature::onInitialize() -> void
{
	coloringInfo_ = sharedParameters_->get<ColoringInfo>("coloringInfo");
	if (coloringInfo_ == nullptr)
	{
		return;
	}

	coloringInfo_->backgroundColors = defaultColors_;
}

auto b3d::renderer::BackgroundColorFeature::getParamsData() -> ParamsData
{
	return { coloringInfo_->backgroundColors };
}

auto b3d::renderer::BackgroundColorFeature::gui() -> void
{
	ImGui::ColorEdit3("Color 1", &coloringInfo_->backgroundColors[0].r);
	ImGui::ColorEdit3("Color 2", &coloringInfo_->backgroundColors[1].r);
}
auto b3d::renderer::BackgroundColorFeature::hasGui() const -> bool
{
	return true;
}
