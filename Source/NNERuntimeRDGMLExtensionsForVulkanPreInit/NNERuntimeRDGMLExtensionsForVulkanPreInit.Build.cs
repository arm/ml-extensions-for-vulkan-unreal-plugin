// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

using UnrealBuildTool;
using System.IO;

public class NNERuntimeRDGMLExtensionsForVulkanPreInit : ModuleRules
{
	public NNERuntimeRDGMLExtensionsForVulkanPreInit( ReadOnlyTargetRules Target ) : base( Target )
	{
		PrivateDependencyModuleNames.AddRange
			(
			new string[] {
				"Core",
				"CoreUObject",
				"VulkanRHI",
				"RHI"
			}
		);

		// So that we can include public headers from VulkanRHI, we need to add the Vulkan include paths
		AddEngineThirdPartyPrivateStaticDependencies(Target, "Vulkan");
	}
}
