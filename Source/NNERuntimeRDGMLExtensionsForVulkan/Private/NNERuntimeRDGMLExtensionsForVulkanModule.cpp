// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "NNERuntimeRDGMLExtensionsForVulkanModule.h"
#include "NNE.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "NNERuntimeRDGMLExtensionsForVulkan.h"
#if WITH_EDITOR
#include "EditorClassUtils.h"
#include "Factories/Factory.h"
#endif
#include "IVulkanDynamicRHI.h"

DEFINE_LOG_CATEGORY(LogNNERuntimeRDGMLExtensionsForVulkan);

/// Attempts to initialize things that we need in order to run inferences using the ML Extensions for Vulkan.
/// This is distinct to things that we need in order to create model data ('compile') a model for later inference.
/// This distinction is important when running the cook commandlet, as that can't run inferences (no RHI) but still
/// needs to compile the model data.
bool InitializeForInference()
{
	if (IsRunningCookCommandlet())
	{
		// If cooking, we won't have an RHI and can't use this plugin. The RHI check below would catch this, but will report
		// an error which will fail the cooking commandlet. Instead we detect cooking separately and log this at a lower
		// severity.
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Log, TEXT("Cooking detected - the ML Extensions for Vulkan NNE Runtime will not be able to run inferences."))
		return false;
	}

	// We need to be using Vulkan to run inferences.
	if (GDynamicRHI == nullptr || GDynamicRHI->GetInterfaceType() != ERHIInterfaceType::Vulkan)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("RHI is not Vulkan. The ML Extensions for Vulkan NNE Runtime will not be able to run inferences."))
		return false;
	}

	// Fetch function pointers. Unfortunately Unreal doesn't expose even the core functions outside of the VulkanRHI module.
	IVulkanDynamicRHI* VulkanRHI = GetIVulkanDynamicRHI();
	bool ErrorGettingFunctions = false;

	auto LoadFunction = [&](void** Dest, const char* Name, bool IsInstanceProc = false)
		{
			*Dest = IsInstanceProc ? VulkanRHI->RHIGetVkInstanceProcAddr(Name) : VulkanRHI->RHIGetVkDeviceProcAddr(Name);
			if (*Dest == nullptr)
			{
				UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Failed to get Vulkan function pointer for '%s'. The ML Extensions for Vulkan NNE Runtime will not be able to run inferences."), *FString(Name));
				ErrorGettingFunctions = true;
			}
		};

	LoadFunction((void**)&vkCreateTensorARM_p, "vkCreateTensorARM");
	LoadFunction((void**)&vkCreateTensorViewARM_p, "vkCreateTensorViewARM");
	LoadFunction((void**)&vkBindTensorMemoryARM_p, "vkBindTensorMemoryARM");
	LoadFunction((void**)&vkCreateDataGraphPipelinesARM_p, "vkCreateDataGraphPipelinesARM");
	LoadFunction((void**)&vkCreateDataGraphPipelineSessionARM_p, "vkCreateDataGraphPipelineSessionARM");
	LoadFunction((void**)&vkCmdDispatchDataGraphARM_p, "vkCmdDispatchDataGraphARM");
	LoadFunction((void**)&vkGetDataGraphPipelineSessionMemoryRequirementsARM_p, "vkGetDataGraphPipelineSessionMemoryRequirementsARM");
	LoadFunction((void**)&vkBindDataGraphPipelineSessionMemoryARM_p, "vkBindDataGraphPipelineSessionMemoryARM");
	LoadFunction((void**)&vkDestroyDataGraphPipelineSessionARM_p, "vkDestroyDataGraphPipelineSessionARM");
	LoadFunction((void**)&vkDestroyTensorARM_p, "vkDestroyTensorARM");
	LoadFunction((void**)&vkDestroyTensorViewARM_p, "vkDestroyTensorViewARM");

	LoadFunction((void**)&vkGetPhysicalDeviceQueueFamilyProperties_p, "vkGetPhysicalDeviceQueueFamilyProperties", true);
	LoadFunction((void**)&vkCreatePipelineLayout_p, "vkCreatePipelineLayout");
	LoadFunction((void**)&vkCreateShaderModule_p, "vkCreateShaderModule");
	LoadFunction((void**)&vkCreateDescriptorSetLayout_p, "vkCreateDescriptorSetLayout");
	LoadFunction((void**)&vkCmdBindPipeline_p, "vkCmdBindPipeline");
	LoadFunction((void**)&vkCreateDescriptorPool_p, "vkCreateDescriptorPool");
	LoadFunction((void**)&vkAllocateDescriptorSets_p, "vkAllocateDescriptorSets");
	LoadFunction((void**)&vkUpdateDescriptorSets_p, "vkUpdateDescriptorSets");
	LoadFunction((void**)&vkCmdBindDescriptorSets_p, "vkCmdBindDescriptorSets");
	LoadFunction((void**)&vkDestroyPipelineLayout_p, "vkDestroyPipelineLayout");
	LoadFunction((void**)&vkDestroyShaderModule_p, "vkDestroyShaderModule");
	LoadFunction((void**)&vkDestroyPipeline_p, "vkDestroyPipeline");
	LoadFunction((void**)&vkDestroyDescriptorSetLayout_p, "vkDestroyDescriptorSetLayout");
	LoadFunction((void**)&vkDestroyDescriptorPool_p, "vkDestroyDescriptorPool");
	LoadFunction((void**)&vkFreeDescriptorSets_p, "vkFreeDescriptorSets");

	if (ErrorGettingFunctions)
	{
		// Error has already been logged by the LoadFunction lambda.
		return false;
	}

	// Check that the VkQueue chosen by Unreal supports data graphs. Unfortuntely we have no way to influence what queue gets chosen,
	// so we have to hope that it picks one that has the support we need.
	VkPhysicalDevice PhysicalDevice = VulkanRHI->RHIGetVkPhysicalDevice();
	uint32 QueueFamilyIndex = VulkanRHI->RHIGetGraphicsQueueFamilyIndex();
	TArray<VkQueueFamilyProperties> QueueFamilyProperties;
	uint32 NumQueueFamilies = QueueFamilyIndex + 1;
	QueueFamilyProperties.AddZeroed(NumQueueFamilies);
	vkGetPhysicalDeviceQueueFamilyProperties_p(PhysicalDevice, &NumQueueFamilies, QueueFamilyProperties.GetData());
	if ((QueueFamilyProperties[QueueFamilyIndex].queueFlags & VK_QUEUE_DATA_GRAPH_BIT_ARM) == 0)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Log, TEXT("Vulkan queue does not support data graphs - the ML Extensions for Vulkan NNE Runtime will not be able to run inferences."))
		return false;
	}

	return true;
}

void FNNERuntimeRDGMLExtensionsForVulkanModule::StartupModule()
{
	// Note that this may fail, but that's fine - we just won't support running inferences. We can still create model data for later inferences
	// which we need when cooking.
	bool SupportsInference = InitializeForInference();

	// Create and register the runtime object with the NNE framework.
	NNERuntimeRDGMLExtensionsForVulkan = NewObject<UNNERuntimeRDGMLExtensionsForVulkan>();
	NNERuntimeRDGMLExtensionsForVulkan->SupportsInference = SupportsInference;
	NNERuntimeRDGMLExtensionsForVulkan->AddToRoot(); // Prevent from being GC'd.
	UE::NNE::RegisterRuntime(NNERuntimeRDGMLExtensionsForVulkan.Get());
}

void FNNERuntimeRDGMLExtensionsForVulkanModule::ShutdownModule()
{
	// Unregister and destroy the runtime object.
	if (NNERuntimeRDGMLExtensionsForVulkan.IsValid())
	{
		UE::NNE::UnregisterRuntime(NNERuntimeRDGMLExtensionsForVulkan.Get());
		NNERuntimeRDGMLExtensionsForVulkan->RemoveFromRoot(); // Allow GC to destroy it.
		NNERuntimeRDGMLExtensionsForVulkan.Reset();
	}
}

IMPLEMENT_MODULE(FNNERuntimeRDGMLExtensionsForVulkanModule, NNERuntimeRDGMLExtensionsForVulkan);
