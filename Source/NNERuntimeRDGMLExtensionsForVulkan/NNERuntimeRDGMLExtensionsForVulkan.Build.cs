// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

using UnrealBuildTool;
using System.IO;

public class NNERuntimeRDGMLExtensionsForVulkan : ModuleRules
{
	public NNERuntimeRDGMLExtensionsForVulkan( ReadOnlyTargetRules Target ) : base( Target )
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PrivateDependencyModuleNames.AddRange
			(
			new string[] {
				"Core",
				"CoreUObject",
				"Engine",
				"NNE",
				"Projects",
				"RHI",
				"RenderCore",
				"VulkanRHI"
			}
		);

        // Include more recent Vulkan headers (which include the ML Extensions for Vulkan) - these will be used 
        // in our module in preference to Unreal's normal Vulkan headers
		string VulkanIncludeDir = Path.GetFullPath(Path.Combine(ModuleDirectory, "..", "ThirdParty", "Vulkan", "Include"));
		PrivateIncludePaths.Add(VulkanIncludeDir);
		PrivateIncludePaths.Add(Path.Combine(VulkanIncludeDir, "vulkan"));

		// We need the VGF library to parse the VGF file.
		PrivateDependencyModuleNames.Add("AIMLSDKVGFLibrary");
		// We need the third-party SPIRV-Tools module at runtime to run shape inference on models.
		PrivateDependencyModuleNames.Add("SPIRVTools");
	}
}
