// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "Modules/ModuleManager.h"
#include "UObject/WeakObjectPtr.h"
#include "Containers/Array.h"
#include "VulkanThirdParty.h"

class UNNERuntimeRDGMLExtensionsForVulkan;

DECLARE_LOG_CATEGORY_EXTERN(LogNNERuntimeRDGMLExtensionsForVulkan, Log, All);

class FNNERuntimeRDGMLExtensionsForVulkanModule : public IModuleInterface
{
private:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

	// Pointer to the singleton NNE runtime object which we create/register with NNE on module startup
	// and unregister/destroy on module shutdown.
	TWeakObjectPtr<UNNERuntimeRDGMLExtensionsForVulkan> NNERuntimeRDGMLExtensionsForVulkan;
};

// Function pointers for Arm extensions.
PFN_vkCreateTensorARM									vkCreateTensorARM_p									 = nullptr;
PFN_vkCreateTensorViewARM								vkCreateTensorViewARM_p								 = nullptr;
PFN_vkBindTensorMemoryARM								vkBindTensorMemoryARM_p								 = nullptr;
PFN_vkCreateDataGraphPipelinesARM						vkCreateDataGraphPipelinesARM_p						 = nullptr;
PFN_vkCreateDataGraphPipelineSessionARM					vkCreateDataGraphPipelineSessionARM_p				 = nullptr;
PFN_vkCmdDispatchDataGraphARM							vkCmdDispatchDataGraphARM_p							 = nullptr;
PFN_vkGetDataGraphPipelineSessionMemoryRequirementsARM	vkGetDataGraphPipelineSessionMemoryRequirementsARM_p = nullptr;
PFN_vkBindDataGraphPipelineSessionMemoryARM				vkBindDataGraphPipelineSessionMemoryARM_p			 = nullptr;
PFN_vkDestroyDataGraphPipelineSessionARM				vkDestroyDataGraphPipelineSessionARM_p				 = nullptr;
PFN_vkDestroyTensorARM									vkDestroyTensorARM_p								 = nullptr;
PFN_vkDestroyTensorViewARM								vkDestroyTensorViewARM_p							 = nullptr;

// Function pointers for core Vulkan functions (unfortunately Unreal doesn't expose these outside of the VulkanRHI module).
PFN_vkGetPhysicalDeviceQueueFamilyProperties            vkGetPhysicalDeviceQueueFamilyProperties_p			 = nullptr;
PFN_vkCreatePipelineLayout								vkCreatePipelineLayout_p							 = nullptr;
PFN_vkCreateShaderModule								vkCreateShaderModule_p								 = nullptr;
PFN_vkCreateDescriptorSetLayout							vkCreateDescriptorSetLayout_p						 = nullptr;
PFN_vkCmdBindPipeline									vkCmdBindPipeline_p									 = nullptr;
PFN_vkCreateDescriptorPool								vkCreateDescriptorPool_p							 = nullptr;
PFN_vkAllocateDescriptorSets							vkAllocateDescriptorSets_p							 = nullptr;
PFN_vkUpdateDescriptorSets								vkUpdateDescriptorSets_p							 = nullptr;
PFN_vkCmdBindDescriptorSets								vkCmdBindDescriptorSets_p							 = nullptr;
PFN_vkDestroyPipelineLayout								vkDestroyPipelineLayout_p							 = nullptr;
PFN_vkDestroyShaderModule								vkDestroyShaderModule_p								 = nullptr;
PFN_vkDestroyPipeline									vkDestroyPipeline_p									 = nullptr;
PFN_vkDestroyDescriptorSetLayout						vkDestroyDescriptorSetLayout_p						 = nullptr;
PFN_vkDestroyDescriptorPool								vkDestroyDescriptorPool_p							 = nullptr;
PFN_vkFreeDescriptorSets								vkFreeDescriptorSets_p								 = nullptr;
