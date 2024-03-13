#pragma once

#include <RenderData.h>
#include "RenderFeature.h"

namespace b3d::renderer
{
	// template <unsigned int S>
	class BackgroundColorFeature final : public RenderFeature
	{
	public:
		explicit BackgroundColorFeature(const std::string& name);

		explicit BackgroundColorFeature(const std::string& name, std::array<ColorRGB, 2> colors);

		auto beginUpdate() -> void override;
		auto onInitialize() -> void override;
		auto gui() -> void override;
		[[nodiscard]] auto hasGui() const -> bool override;

		struct ParamsData
		{
			std::array<ColorRGBA, 2> colors;
		};

		[[nodiscard]] auto getParamsData() -> ParamsData;

	private:
		std::array<ColorRGBA, 2> defaultColors_;

		ColoringInfo* coloringInfo_{ nullptr };
	};

} // namespace b3d::renderer
