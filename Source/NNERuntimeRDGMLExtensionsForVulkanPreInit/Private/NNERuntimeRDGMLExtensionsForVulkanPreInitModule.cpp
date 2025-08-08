// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "NNERuntimeRDGMLExtensionsForVulkanPreInitModule.h"
#include "IVulkanDynamicRHI.h"

void FNNERuntimeRDGMLExtensionsForVulkanPreInitModule::StartupModule()
{
	// We need to enable the ML Extensions for Vulkan in the Vulkan RHI before it initializes.
	// That's why this code is in a separate module to the main module, as it loads earlier.
	TArray<const ANSICHAR*> Extensions;
	Extensions.Add("VK_ARM_tensors");
	Extensions.Add("VK_ARM_data_graph");
	// These are dependencies of the above extensions.
	Extensions.Add("VK_KHR_maintenance5");
	Extensions.Add("VK_KHR_deferred_host_operations");

	IVulkanDynamicRHI::AddEnabledDeviceExtensionsAndLayers(Extensions, TArrayView<const ANSICHAR* const>());
}

void FNNERuntimeRDGMLExtensionsForVulkanPreInitModule::ShutdownModule()
{
}

IMPLEMENT_MODULE(FNNERuntimeRDGMLExtensionsForVulkanPreInitModule, NNERuntimeRDGMLExtensionsForVulkanPreInit);