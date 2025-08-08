// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.IO;
using UnrealBuildTool;

public class SPIRVTools : ModuleRules
{
	public SPIRVTools(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;

		string PlatformDir = Target.Platform.ToString(); // e.g. Win64, Linux
		PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "Include"));

		string LibDirPath = Path.Combine(ModuleDirectory, "Lib", PlatformDir);

		PublicAdditionalLibraries.Add(Path.Combine(LibDirPath, "SPIRV-Tools.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(LibDirPath, "SPIRV-Tools-opt.lib"));
	}
}