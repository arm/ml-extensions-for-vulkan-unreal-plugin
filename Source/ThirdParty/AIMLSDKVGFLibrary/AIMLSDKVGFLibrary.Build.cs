// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.IO;
using UnrealBuildTool;
public class AIMLSDKVGFLibrary : ModuleRules
{
	public AIMLSDKVGFLibrary(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;
		string PlatformDir = Target.Platform.ToString(); // e.g. Win64, Linux
		PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "Include"));
		string LibDirPath = Path.Combine(ModuleDirectory, "Lib", PlatformDir);
		string BinariesDirPath = Path.Combine(PluginDirectory, "Binaries", "ThirdParty", "AIMLSDKVGFLibrary", PlatformDir);
		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			PublicAdditionalLibraries.Add(Path.Combine(LibDirPath, "vgf.lib"));
		}
        else 
        {
            throw new BuildException("Platform {0} doesn't support AIMLSDKVGFLibrary libraries.", Target.Platform);
        }
	}
}