// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

using UnrealBuildTool;
using System.IO;

public class NNERuntimeRDGMLExtensionsForVulkanEditor : ModuleRules
{
	public NNERuntimeRDGMLExtensionsForVulkanEditor( ReadOnlyTargetRules Target ) : base( Target )
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PrivateDependencyModuleNames.AddRange
			(
			new string[] {
				"Core",
				"CoreUObject",
				"UnrealEd",
				"NNE",
				"Engine"
			}
		);
	}
}
