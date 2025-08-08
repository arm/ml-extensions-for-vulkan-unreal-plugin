// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "NNERuntimeRDGMLExtensionsForVulkanModel.h"
#include "NNERuntimeRDGMLExtensionsForVulkanModule.h"
#include "NNERuntimeRDGMLExtensionsForVulkanShapeInference.h"
#include "RenderGraphBuilder.h"
#include "NNERuntimeRDGMLExtensionsForVulkan.h"
#include "Algo/Accumulate.h"
#include "Algo/Transform.h"

class FVulkanDevice; // Forward declaration needed for VulkanUtil.h
#include "VulkanUtil.h"

#include "vgf/decoder.h" // The VGF parser from the ML SDK for Vulkan

// The max number of executions that can be queued up (on the GPU) for each model instance.
const uint32_t MAX_CONCURRENT_EXECUTIONS_PER_INSTANCE = 10;

uint32 GetTypeHash(const UE::NNE::FTensorShape& Shape)
{
	return GetArrayHash(Shape.GetData().GetData(), Shape.GetData().Num());
}

uint32 GetTypeHash(const TArray<UE::NNE::FTensorShape>& ArrayOfShapes)
{
	uint32 Hash = 0;
	for (const UE::NNE::FTensorShape& S : ArrayOfShapes)
	{
		Hash = HashCombineFast(Hash, GetTypeHash(S));
	}
	return Hash;
}

namespace Private
{

ENNETensorDataType VKFormatToNNETensorDataType(mlsdk_vk_format VKFormat)
{
	// The VGF format enum is just the regular Vulkan VkFormat.
	switch ((VkFormat)VKFormat)
	{
	case VK_FORMAT_R32_SFLOAT:
		return ENNETensorDataType::Float;
	case VK_FORMAT_R8_SINT:
		return ENNETensorDataType::Int8;
	default:
		return ENNETensorDataType::None;
	}
}

size_t GetNumBytesPerElement(mlsdk_vk_format VKFormat)
{
	// The VGF format enum is just the regular Vulkan VkFormat.
	switch ((VkFormat)VKFormat)
	{
	case VK_FORMAT_R32_SFLOAT:
		return 4;
	case VK_FORMAT_R8_SINT:
		return 1;
	default:
		return 0;
	}
}

}

TSharedPtr<FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped> FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped::Create(const TSharedPtr<UE::NNE::FSharedModelData>& InModelData)
{
	TSharedPtr<FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped> Result(new FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped());
	Result->SharedModelData = InModelData; // Keep a reference to this alive, as we'll use it when creating shaped models later.

	// Skip past the GUID and version (which have already been validated by UNNERuntimeRDGMLExtensionsForVulkan::CreateModelRDG) to get to the raw VGF data.
	TConstArrayView<uint8> VgfBuffer =
		InModelData->GetView().RightChop(sizeof(UNNERuntimeRDGMLExtensionsForVulkan::ModelDataGUID) + sizeof(UNNERuntimeRDGMLExtensionsForVulkan::ModelDataVersion));

	// Parse VGF header which contains details of other sections in the file.
	TArray<uint8_t> HeaderDecoderMemory;
	HeaderDecoderMemory.AddUninitialized(mlsdk_decoder_header_decoder_mem_reqs());
	mlsdk_decoder_header_decoder* HeaderDecoder = mlsdk_decoder_create_header_decoder(VgfBuffer.GetData(), HeaderDecoderMemory.GetData());
	if (!mlsdk_decoder_is_header_valid(HeaderDecoder))
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Invalid VGF header."));
		return nullptr;
	}
	if (!mlsdk_decoder_is_header_compatible(HeaderDecoder))
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Incompatible VGF header."));
		return nullptr;
	}

	// Create decoder objects for each section in the VGF that we care about:
	//		Module Table: 
	//			Each module is either a compute shader or a data graph.
	//			The order of these is arbitrary and there is a further information in the VGF that describes how to run these.
	//		Model Resource Table:
	//			This is a list of tensor descriptions (data formats, size etc.) which is indexed
	//			into by other fields in the VGF.
	//		Model Sequence:
	//			This defines the order that the modules should be executed in as well as their inputs and outputs.
	//		Constant table:
	//			Contains the raw constant data for all constant tensors used in the model.
	mlsdk_decoder_vgf_section_info SectionInfos[4];
	for (mlsdk_decoder_section SectionType = mlsdk_decoder_section_modules; SectionType <= mlsdk_decoder_section_constants;
		SectionType = mlsdk_decoder_section(SectionType + 1))
	{
		mlsdk_decoder_get_header_section_info(HeaderDecoder, SectionType, &SectionInfos[SectionType]);
		if (SectionInfos[SectionType].offset + SectionInfos[SectionType].size > VgfBuffer.Num())
		{
			UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Corrupt VGF header (section out of bounds)."));
			return nullptr;
		}
	}
	TArray<uint8_t> ModuleTableDecoderMemory;
	TArray<uint8_t> ModelResourceTableDecoderMemory;
	TArray<uint8_t> ModelSequenceDecoderMemory;
	TArray<uint8_t> ConstantTableDecoderMemory;
	ModuleTableDecoderMemory.AddUninitialized(mlsdk_decoder_module_table_decoder_mem_reqs());
	ModelResourceTableDecoderMemory.AddUninitialized(mlsdk_decoder_model_resource_table_decoder_mem_reqs());
	ModelSequenceDecoderMemory.AddUninitialized(mlsdk_decoder_model_sequence_decoder_mem_reqs());
	ConstantTableDecoderMemory.AddUninitialized(mlsdk_decoder_constant_table_decoder_mem_reqs());
	mlsdk_decoder_module_table_decoder* ModuleTableDecoder =
		mlsdk_decoder_create_module_table_decoder(VgfBuffer.GetData() + SectionInfos[mlsdk_decoder_section_modules].offset, ModuleTableDecoderMemory.GetData());
	mlsdk_decoder_model_resource_table_decoder* ModelResourceTableDecoder =
		mlsdk_decoder_create_model_resource_table_decoder(VgfBuffer.GetData() + SectionInfos[mlsdk_decoder_section_resources].offset, ModelResourceTableDecoderMemory.GetData());
	mlsdk_decoder_model_sequence_decoder* ModelSequenceDecoder =
		mlsdk_decoder_create_model_sequence_decoder(VgfBuffer.GetData() + SectionInfos[mlsdk_decoder_section_model_sequence].offset, ModelSequenceDecoderMemory.GetData());
	mlsdk_decoder_constant_table_decoder* ConstantTableDecoder =
		mlsdk_decoder_create_constant_table_decoder(VgfBuffer.GetData() + SectionInfos[mlsdk_decoder_section_constants].offset, ConstantTableDecoderMemory.GetData());

	// Create FTensorShapes, VkTensorDescriptionARM etc. for each resource in the model resource table. We will look these up later.
	// Note that some data in this struct might be missing, depending on the resource. For example not all tensors will have a concrete shape.
	struct FResourceDesc
	{
		ENNETensorDataType NNEDataType = ENNETensorDataType::None;
		VkTensorDescriptionARM TensorDescription = {};
		TArray<int64_t> TensorShapeRawS64; // Storage for the pointer in VkTensorDescriptionARM.
		UE::NNE::FSymbolicTensorShape SymbolicTensorShape;
		// Lookup from the index in the VGF model resource table to our renumbered IDs.
		// Not all resources have a TensorId though, so this can be -1 (e.g. for constants).
		int32 TensorId = -1;
	};
	const size_t NumModelResourceTableEntries = mlsdk_decoder_get_model_resource_table_num_entries(ModelResourceTableDecoder);
	TArray<FResourceDesc> ResourceDescs;
	ResourceDescs.Reserve(NumModelResourceTableEntries);
	for (int ResourceIdx = 0; ResourceIdx < NumModelResourceTableEntries; ++ResourceIdx)
	{
		FResourceDesc ResourceDesc = {};

		mlsdk_vk_format VKFormat = mlsdk_decoder_get_vk_format(ModelResourceTableDecoder, ResourceIdx);
		ResourceDesc.NNEDataType = Private::VKFormatToNNETensorDataType(VKFormat);

		ResourceDesc.TensorDescription.sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM;
		ResourceDesc.TensorDescription.tiling = VK_TENSOR_TILING_LINEAR_ARM;
		ResourceDesc.TensorDescription.usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
		ResourceDesc.TensorDescription.format = static_cast<VkFormat>(VKFormat);

		mlsdk_decoder_tensor_dimensions DimsRaw;
		mlsdk_decoder_model_resource_table_get_tensor_shape(ModelResourceTableDecoder, ResourceIdx, &DimsRaw);

		ResourceDesc.TensorShapeRawS64.Reserve(DimsRaw.size);
		TArray<int32_t> DimsS32;
		DimsS32.Reserve(DimsRaw.size);
		for (int I = 0; I < DimsRaw.size; ++I)
		{
			int64_t x = DimsRaw.data[I];
			if (x <= 0)
			{
				// Negative values indicate that this dimension isn't specified in the model, and will need to determined
				// by shape inference which can be done once the user calls SetInputTensorShapes to set concrete shapes.
				x = -1; // Normalize any unspecified dimension to -1, for consistency with NNE tensor shape types.
			}
			ResourceDesc.TensorShapeRawS64.Push(x);
			DimsS32.Push(int32_t(x));
		}

		ResourceDesc.SymbolicTensorShape = UE::NNE::FSymbolicTensorShape::Make(DimsS32);

		ResourceDesc.TensorDescription.dimensionCount = DimsRaw.size;
		ResourceDesc.TensorDescription.pDimensions = ResourceDesc.TensorShapeRawS64.GetData();

		mlsdk_decoder_tensor_dimensions StridesRaw;
		mlsdk_decoder_model_resource_table_get_tensor_strides(ModelResourceTableDecoder, ResourceIdx, &StridesRaw);
		if (StridesRaw.size > 0)
		{
			UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Strides not supported."));
			return nullptr;
		}

		mlsdk_decoder_mrt_category Category = mlsdk_decoder_model_resource_table_get_category(ModelResourceTableDecoder, ResourceIdx);
		if (Category == mlsdk_decoder_mrt_category::mlsdk_decoder_mrt_category_input ||
			Category == mlsdk_decoder_mrt_category::mlsdk_decoder_mrt_category_output ||
			Category == mlsdk_decoder_mrt_category::mlsdk_decoder_mrt_category_intermediate)
		{
			// We need to store info about these types of buffers outside of this creation function,
			// so that we can allocate intermediates and match up inputs/outputs at inference time.
			FTensorInfoUnshaped Info = {};
			Info.VulkanDesc = ResourceDesc.TensorDescription;
			Info.VulkanDesc.pDimensions = nullptr; // As the shape may have unspecified dimensions (e.g. -1) at this point, don't bother to store it. It will be inferred through shape inference later.
			// The ModelInput/OutputIdx fields will be filled in later.

			// Assign this tensor the next (consecutive) ID.
			uint32 TensorId = Result->TensorInfosUnshaped.Num();
			Result->TensorInfosUnshaped.Add(MoveTemp(Info)); // Move is important here, so that the pDimensions pointer remains valid.
			ResourceDesc.TensorId = TensorId; // So that we can lookup from resource idx to TensorId later.
		}

		ResourceDescs.Add(MoveTemp(ResourceDesc)); // Move is important here, so that the pDimensions pointer remains valid.
	}

	// Check which tensors are model inputs/output and update the TensorInfos for these.
	auto ProcessModelEndpoints = [&](const char* NamePrefix, mlsdk_decoder_binding_slots_handle Bindings,
		TArray<UE::NNE::FTensorDesc>& OutTensorDescs, int32 FTensorInfoUnshaped::* PointerToTensorInfoInputOrOutputIdx)
		{
			const size_t NumBindings = mlsdk_decoder_binding_slot_size(ModelSequenceDecoder, Bindings);
			OutTensorDescs.AddZeroed(NumBindings);
			for (int Idx = 0; Idx < NumBindings; ++Idx)
			{
				uint32_t ResourceIndex = mlsdk_decoder_binding_slot_mrt_index(ModelSequenceDecoder, Bindings, Idx);
				if (ResourceIndex >= NumModelResourceTableEntries)
				{
					UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Corrupt VGF (resource index out of bounds)."));
					return false;
				}

				// Which model input/output this is (presumably the order of inputs/outputs in the VGF is not guaranteed to match the model input/output order)
				uint32 ModelIdx = Idx;

				const UE::NNE::FSymbolicTensorShape SymbolicShape = ResourceDescs[ResourceIndex].SymbolicTensorShape;
				const UE::NNE::FTensorDesc TensorDesc = UE::NNE::FTensorDesc::Make(NamePrefix + FString::FromInt(Idx), SymbolicShape, ResourceDescs[ResourceIndex].NNEDataType);

				OutTensorDescs[ModelIdx] = TensorDesc;

				int32 TensorId = ResourceDescs[ResourceIndex].TensorId;
				if (TensorId == -1)
				{
					UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Invalid VGF (model input or output has incorrect resource type)."));
					return false;
				}
				Result->TensorInfosUnshaped[TensorId].*PointerToTensorInfoInputOrOutputIdx = ModelIdx;
			}

			return true;
		};

	// Inputs
	if (!ProcessModelEndpoints("Input", mlsdk_decoder_model_sequence_get_input_binding_slot(ModelSequenceDecoder),
		Result->InputSymbolicTensors, &FTensorInfoUnshaped::ModelInputIdx))
	{
		return nullptr; // Error already logged by ProcessModelEndpoints.
	}
	// Outputs
	if (!ProcessModelEndpoints("Output", mlsdk_decoder_model_sequence_get_output_binding_slot(ModelSequenceDecoder),
		Result->OutputSymbolicTensors, &FTensorInfoUnshaped::ModelOutputIdx))
	{
		return nullptr; // Error already logged by ProcessModelEndpoints.
	}

	// Loop over model sequence table, which is a list of 'segments' describing which modules (see above) to run in what order
	// and what inputs/outputs they should have. This order handles any dependencies between modules.
	// Create and store the Vulkan pipelines etc. that will be needed to run each segment (but only ones that can be shared between instances)
	const size_t NumModelSequenceTableEntries = mlsdk_decoder_get_model_sequence_table_size(ModelSequenceDecoder);
	for (int ModelSequenceTableIdx = 0; ModelSequenceTableIdx < NumModelSequenceTableEntries; ++ModelSequenceTableIdx)
	{
		FSegmentUnshaped Segment = {};

		const char* SegmentNameRaw = mlsdk_decoder_model_sequence_get_segment_name(ModelSequenceDecoder, ModelSequenceTableIdx);
		Segment.Name = SegmentNameRaw;
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Verbose, TEXT("Parsing segment %s"), *Segment.Name);

		int32_t ModuleIndex = mlsdk_decoder_model_sequence_get_segment_module_index(ModelSequenceDecoder, ModelSequenceTableIdx);

		mlsdk_decoder_module_type SegmentType = mlsdk_decoder_model_sequence_get_segment_type(ModelSequenceDecoder, ModelSequenceTableIdx);
		if (SegmentType != mlsdk_decoder_module_type::mlsdk_decoder_module_type_graph)
		{
			UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Non-graph segments not supported."));
			return nullptr;
		}

		mlsdk_decoder_binding_slots_handle SegmentInputBindings = mlsdk_decoder_model_sequence_get_segment_input_binding_slot(ModelSequenceDecoder, ModelSequenceTableIdx);
		const size_t NumSegmentInputBindings = mlsdk_decoder_binding_slot_size(ModelSequenceDecoder, SegmentInputBindings);

		mlsdk_decoder_binding_slots_handle SegmentOutputBindings = mlsdk_decoder_model_sequence_get_segment_output_binding_slot(ModelSequenceDecoder, ModelSequenceTableIdx);
		const size_t NumSegmentOutputBindings = mlsdk_decoder_binding_slot_size(ModelSequenceDecoder, SegmentOutputBindings);

		TArray<VkDescriptorSetLayoutBinding> DescriptorSetLayoutBindings;
		DescriptorSetLayoutBindings.Reserve(NumSegmentInputBindings + NumSegmentOutputBindings);

		// Gather graph pipeline bindings for inputs and outputs of this segment.
		auto ProcessSegmentEndpoints = [&](mlsdk_decoder_binding_slots_handle Bindings, size_t NumBindings, FSegmentUnshaped::FBinding::EBindingKind BindingKind) {
			for (int I = 0; I < NumBindings; ++I)
			{
				uint32_t ResourceIndex = mlsdk_decoder_binding_slot_mrt_index(ModelSequenceDecoder, Bindings, I);
				if (ResourceIndex >= NumModelResourceTableEntries)
				{
					UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Corrupt VGF (resource index out of bounds)."));
					return false;
				}

				VkDescriptorSetLayoutBinding LayoutBinding = {};
				LayoutBinding.binding = mlsdk_decoder_binding_slot_binding_id(ModelSequenceDecoder, Bindings, I);
				LayoutBinding.descriptorCount = 1;
				LayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
				LayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;
				DescriptorSetLayoutBindings.Add(LayoutBinding);

				FSegmentUnshaped::FBinding OurBinding = {};
				OurBinding.BindingKind = BindingKind;
				OurBinding.VulkanBindingIdx = LayoutBinding.binding;
				int32 TensorId = ResourceDescs[ResourceIndex].TensorId;
				if (TensorId == -1)
				{
					UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Invalid VGF (segment input or output has incorrect resource type)."));
					return false;
				}
				OurBinding.TensorId = TensorId;

				Segment.Bindings.Add(OurBinding);
			}
			return true;
			};

		// Inputs for this segment.
		if (!ProcessSegmentEndpoints(SegmentInputBindings, NumSegmentInputBindings, FSegmentUnshaped::FBinding::EBindingKind::Input))
		{
			return nullptr; // Error already logged by ProcessSegmentEndpoints.
		}
		// Outputs for this segment.
		if (!ProcessSegmentEndpoints(SegmentOutputBindings, NumSegmentOutputBindings, FSegmentUnshaped::FBinding::EBindingKind::Output))
		{
			return nullptr; // Error already logged by ProcessSegmentEndpoints.
		}

		const size_t NumDescriptorSets = mlsdk_decoder_model_sequence_get_segment_descriptorset_info_size(ModelSequenceDecoder, ModelSequenceTableIdx);
		if (NumDescriptorSets != 1)
		{
			// These are probably only needed for compute segments (which we don't support yet), and for graph segments we have all the info we need
			// in the segment input/output bindings, so we just do a basic sanity check on this.
			UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Descriptor sets count unexpected."));
			return nullptr;
		}

		mlsdk_decoder_push_constant_ranges_handle PushConstantsRanges = mlsdk_decoder_model_sequence_get_segment_push_constant_range(ModelSequenceDecoder, ModelSequenceTableIdx);
		const size_t NumPushConstantRanges = mlsdk_decoder_get_push_constant_ranges_size(ModelSequenceDecoder, PushConstantsRanges);
		if (NumPushConstantRanges != 0)
		{
			// These are probably intended to be used for compute segments, but we don't support these yet.
			UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Push constants not supported."));
			return nullptr;
		}

		// Constants for this segment.
		const size_t NumModelConstants = mlsdk_decoder_get_constant_table_num_entries(ConstantTableDecoder);
		mlsdk_decoder_constant_indexes ConstantIndexes;
		mlsdk_decoder_model_sequence_get_segment_constant_indexes(ModelSequenceDecoder, ModelSequenceTableIdx, &ConstantIndexes);
		Segment.ConstantInfos.Reserve(ConstantIndexes.size);
		for (int ConstantIdxWithinSegment = 0; ConstantIdxWithinSegment < ConstantIndexes.size; ++ConstantIdxWithinSegment)
		{
			int ModelConstantIdx = ConstantIndexes.data[ConstantIdxWithinSegment];
			if (ModelConstantIdx > NumModelConstants)
			{
				UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Corrupt VGF (segment constant idx out of bounds)."));
				return nullptr;
			}

			uint32_t ResourceIndex = mlsdk_decoder_constant_table_get_mrt_index(ConstantTableDecoder, ModelConstantIdx);
			if (ResourceIndex >= NumModelResourceTableEntries)
			{
				UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Corrupt VGF (constant resource idx out of bounds)."));
				return nullptr;
			}

			mlsdk_decoder_constant_data ConstantData;
			mlsdk_decoder_constant_table_get_data(ConstantTableDecoder, ModelConstantIdx, &ConstantData);

			FSegmentUnshaped::FConstantInfo& ConstantInfo = Segment.ConstantInfos.AddZeroed_GetRef();

			ConstantInfo.TensorDescription = ResourceDescs[ResourceIndex].TensorDescription;
			ConstantInfo.TensorDimensions = ResourceDescs[ResourceIndex].TensorShapeRawS64;
			// Important to update the pointer in TensorDescription.pDimensions to a copy of the data which will live alongside it.
			ConstantInfo.TensorDescription.pDimensions = ConstantInfo.TensorDimensions.GetData();

			ConstantInfo.DataGraphPipelineConstant.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM;
			ConstantInfo.DataGraphPipelineConstant.id = ConstantIdxWithinSegment;
			ConstantInfo.DataGraphPipelineConstant.pNext = &ConstantInfo.TensorDescription;
			ConstantInfo.DataGraphPipelineConstant.pConstantData = ConstantData.data;
		}

		mlsdk_decoder_module_type ModuleType = mlsdk_decoder_get_module_type(ModuleTableDecoder, ModuleIndex);
		if (ModuleType != mlsdk_decoder_module_type_graph)
		{
			UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Non-graph modules not supported."));
			return nullptr;
		}

		mlsdk_decoder_spirv_code SPIRVCode;
		mlsdk_decoder_get_module_code(ModuleTableDecoder, ModuleIndex, &SPIRVCode);
		if (SPIRVCode.code == nullptr || SPIRVCode.words == 0)
		{
			UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Missing SPIRV code for module."));
			return nullptr;
		}
		Segment.SPIRVCode = TConstArrayView<uint32_t>(SPIRVCode.code, SPIRVCode.words);

		Segment.SPIRVEntryPoint = mlsdk_decoder_get_module_entry_point(ModuleTableDecoder, ModuleIndex);

		// Run the Vulkan resource creation functions on the RHI thread and wait for them to complete.
		FEvent* RenderThreadDoneEvent = FGenericPlatformProcess::GetSynchEventFromPool(true);
		ENQUEUE_RENDER_COMMAND(NNERuntimeRDGMLExtensionsForVulkanModel_CreateSegment)([&](FRHICommandListImmediate& RHICmdList) {
			RHICmdList.EnqueueLambda([&](FRHICommandListImmediate& RHICmdList) {
				VkDevice Device = GetIVulkanDynamicRHI()->RHIGetVkDevice();
				const VkAllocationCallbacks* Allocator = GetIVulkanDynamicRHI()->RHIGetVkAllocationCallbacks();

				// Descriptor set layout.
				VkDescriptorSetLayoutCreateInfo GraphDescriptorSetLayoutCreateInfo = {};
				GraphDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				GraphDescriptorSetLayoutCreateInfo.bindingCount = DescriptorSetLayoutBindings.Num();
				GraphDescriptorSetLayoutCreateInfo.pBindings = DescriptorSetLayoutBindings.GetData();
				VERIFYVULKANRESULT(vkCreateDescriptorSetLayout_p(Device, &GraphDescriptorSetLayoutCreateInfo, Allocator, &Segment.DescriptorSetLayout));

				// Graph pipeline layout.
				VkPipelineLayoutCreateInfo PipelineLayoutCreateInfo = {};
				PipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				PipelineLayoutCreateInfo.setLayoutCount = 1;
				PipelineLayoutCreateInfo.pSetLayouts = &Segment.DescriptorSetLayout;
				VERIFYVULKANRESULT(vkCreatePipelineLayout_p(Device, &PipelineLayoutCreateInfo, Allocator, &Segment.PipelineLayout));
			});
			RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
			RenderThreadDoneEvent->Trigger();
		});

		RenderThreadDoneEvent->Wait();
		FGenericPlatformProcess::ReturnSynchEventToPool(RenderThreadDoneEvent);

		Result->SegmentsUnshaped.Add(MoveTemp(Segment));
	}

	return Result;
}

FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped::FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped()
{
	// All initialisation logic is in the static Create function (above).
}

FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped::~FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped()
{
	// Destroy Vulkan resources on the RHI thread and wait for that to finish.
	FEvent* RenderThreadDoneEvent = FGenericPlatformProcess::GetSynchEventFromPool(true);
	ENQUEUE_RENDER_COMMAND(NNERuntimeRDGMLExtensionsForVulkanModel_DestroySegments)([&](FRHICommandListImmediate& RHICmdList) {
		RHICmdList.EnqueueLambda([&](FRHICommandListImmediate& RHICmdList) {
			VkDevice Device = GetIVulkanDynamicRHI()->RHIGetVkDevice();
			const VkAllocationCallbacks* Allocator = GetIVulkanDynamicRHI()->RHIGetVkAllocationCallbacks();

			for (FSegmentUnshaped& S : SegmentsUnshaped)
			{
				vkDestroyPipelineLayout_p(Device, S.PipelineLayout, Allocator);
				vkDestroyDescriptorSetLayout_p(Device, S.DescriptorSetLayout, Allocator);
			}

		});
		RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
		RenderThreadDoneEvent->Trigger();
	});

	RenderThreadDoneEvent->Wait();
	FGenericPlatformProcess::ReturnSynchEventToPool(RenderThreadDoneEvent);
}

TSharedPtr<UE::NNE::IModelInstanceRDG> FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped::CreateModelInstanceRDG()
{
	// We can't initialize very much of the model instance yet, because we don't know the concrete tensor shapes 
	// until SetInputTensorShapes is called.
	TSharedPtr<FNNERuntimeRDGMLExtensionsForVulkanModelInstance> Result(new FNNERuntimeRDGMLExtensionsForVulkanModelInstance());
	Result->ParentModelUnshaped = this->AsShared();

	// Create vulkan resources for this instance, using the common resources from the parent model.
	// Run the Vulkan resource creation functions on the RHI thread and wait for them to complete.
	FEvent* RenderThreadDoneEvent = FGenericPlatformProcess::GetSynchEventFromPool(true);
	ENQUEUE_RENDER_COMMAND(NNERuntimeRDGMLExtensionsForVulkanModel_DestroySegments)([&](FRHICommandListImmediate& RHICmdList) {
		RHICmdList.EnqueueLambda([&](FRHICommandListImmediate& RHICmdList) {
			VkDevice Device = GetIVulkanDynamicRHI()->RHIGetVkDevice();
			const VkAllocationCallbacks* Allocator = GetIVulkanDynamicRHI()->RHIGetVkAllocationCallbacks();

			uint32 NumDescriptors = 0;
			for (FSegmentUnshaped& Segment : SegmentsUnshaped)
			{
				// Sum up the total number of descriptors that we will need for all segments.
				NumDescriptors += Segment.Bindings.Num();
			}

			// Create descriptor pool to use for this instance. We could create one of these in the parent model, but then we wouldn't know
			// how big the pool should be as we don't know how many instances will be created.
			VkDescriptorPoolCreateInfo DescriptorPoolCreateInfo = {};
			DescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			DescriptorPoolCreateInfo.maxSets = SegmentsUnshaped.Num() * MAX_CONCURRENT_EXECUTIONS_PER_INSTANCE;
			DescriptorPoolCreateInfo.poolSizeCount = 1;
			DescriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
			VkDescriptorPoolSize PoolSize = {};
			PoolSize.type = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
			PoolSize.descriptorCount = NumDescriptors * MAX_CONCURRENT_EXECUTIONS_PER_INSTANCE;
			DescriptorPoolCreateInfo.pPoolSizes = &PoolSize;
			VERIFYVULKANRESULT(vkCreateDescriptorPool_p(Device, &DescriptorPoolCreateInfo, Allocator, &Result->DescriptorPool));
		});

		RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
		RenderThreadDoneEvent->Trigger();
	});

	RenderThreadDoneEvent->Wait();
	FGenericPlatformProcess::ReturnSynchEventToPool(RenderThreadDoneEvent);

	return Result;
}

TSharedPtr<FNNERuntimeRDGMLExtensionsForVulkanModelShaped> FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped::FindOrCreateShapedModel(TConstArrayView<UE::NNE::FTensorShape> ModelInputShapes)
{
	// Check cache
	TWeakPtr<FNNERuntimeRDGMLExtensionsForVulkanModelShaped>* CacheHit = ShapedModels.Find(TArray<UE::NNE::FTensorShape>(ModelInputShapes));
	if (CacheHit != nullptr && CacheHit->IsValid()) // Note we also need to check that the weak pointer is still alive.
	{
		return CacheHit->Pin();
	}

	// No cache hit - create from scratch and insert into cache.
	TSharedPtr<FNNERuntimeRDGMLExtensionsForVulkanModelShaped> ShapedModel(new FNNERuntimeRDGMLExtensionsForVulkanModelShaped());
	ShapedModel->ParentModelUnshaped = this->AsShared();

	// Run shape inference over the whole VGF, starting from the inputs and working our way through the graph to the outputs.
	ShapedModel->InputTensorShapes = ModelInputShapes;
	ShapedModel->SegmentsShaped.Reserve(SegmentsUnshaped.Num());
	ShapedModel->TensorInfosShaped.Reserve(TensorInfosUnshaped.Num());
	// Start off with copying all the unshaped tensor infos as-is. We will replace the contents with concrete shapes
	// as we go.
	for (const FTensorInfoUnshaped& UnshapedTensorInfo : TensorInfosUnshaped)
	{
		FNNERuntimeRDGMLExtensionsForVulkanModelShaped::FTensorInfoShaped ShapedTensorInfo;
		ShapedTensorInfo.VulkanDesc = UnshapedTensorInfo.VulkanDesc;
		if (UnshapedTensorInfo.ModelInputIdx != -1)
		{
			// This is a model input, so the concrete shape is provided directly (no shape inference necessary).
			ShapedTensorInfo.ShapeRawS64 = ModelInputShapes[UnshapedTensorInfo.ModelInputIdx].GetData();
		}
		ShapedTensorInfo.VulkanDesc.pDimensions = ShapedTensorInfo.ShapeRawS64.GetData(); // Important to update the VkTensorDescription as the array data may have changed!
		// .NumBytes is filled in later, once we know all the tensor shapes.
		ShapedModel->TensorInfosShaped.Add(MoveTemp(ShapedTensorInfo));
	}



	for (int S = 0; S < SegmentsUnshaped.Num(); ++S)
	{
		const FSegmentUnshaped& SegmentUnshaped = SegmentsUnshaped[S];

		FNNERuntimeRDGMLExtensionsForVulkanModelShaped::FSegmentShaped SegmentShaped;
		// For now we only support shape inference for SPIR-V segments (not compute segments)
		// Map of input shapes for this segment.
		TMap<TPair<uint32_t, uint32_t>, TArray<int64_t>> SegmentInputShapes;
		for (int B = 0; B < SegmentUnshaped.Bindings.Num(); ++B)
		{
			if (SegmentUnshaped.Bindings[B].BindingKind == FSegmentUnshaped::FBinding::EBindingKind::Input)
			{
				uint32_t DescriptorSet = 0; // We assume all bindings are in a single descriptor set.
				uint32_t VulkanBindingIdx = SegmentUnshaped.Bindings[B].VulkanBindingIdx;
				SegmentInputShapes.Add({ DescriptorSet , VulkanBindingIdx }, ShapedModel->TensorInfosShaped[SegmentUnshaped.Bindings[B].TensorId].ShapeRawS64);
			}
		}

		// Run shape inference using SPIRV-Tools.
		ShapeInferenceResults ShapeInferenceResults = RunShapeInference(SegmentUnshaped.SPIRVCode, SegmentInputShapes);

		if (!ShapeInferenceResults.Success)
		{
			UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Shape inference failed"));
			return nullptr;
		}

		for (int B = 0; B < SegmentUnshaped.Bindings.Num(); ++B)
		{
			if (SegmentUnshaped.Bindings[B].BindingKind == FSegmentUnshaped::FBinding::EBindingKind::Output)
			{
				uint32_t DescriptorSet = 0; // We assume all bindings are in a single descriptor set.
				uint32_t VulkanBindingIdx = SegmentUnshaped.Bindings[B].VulkanBindingIdx;
				FNNERuntimeRDGMLExtensionsForVulkanModelShaped::FTensorInfoShaped& TensorInfoShaped = ShapedModel->TensorInfosShaped[SegmentUnshaped.Bindings[B].TensorId];
				TensorInfoShaped.ShapeRawS64 = *ShapeInferenceResults.OutputShapes.Find(TPair<uint32_t, uint32_t>{ DescriptorSet, VulkanBindingIdx });
				TensorInfoShaped.VulkanDesc.pDimensions = TensorInfoShaped.ShapeRawS64.GetData(); // Important to update the VkTensorDescription as the array data will have changed!
			}
		}

		// Now that we have the concrete tensor shapes for this segment, we can create the Vulkan pipeline etc.
		TArray<VkDataGraphPipelineConstantARM> DataGraphPipelineConstants;
		Algo::Transform(SegmentUnshaped.ConstantInfos, DataGraphPipelineConstants, [](const auto& x) { return x.DataGraphPipelineConstant; });

		TArray<VkDataGraphPipelineResourceInfoARM> DataGraphPipelineResourcesInfos;
		DataGraphPipelineResourcesInfos.Reserve(SegmentUnshaped.Bindings.Num());
		for (int B = 0; B < SegmentUnshaped.Bindings.Num(); ++B)
		{
			const FSegmentUnshaped::FBinding& Binding = SegmentUnshaped.Bindings[B];

			VkDataGraphPipelineResourceInfoARM ResourceInfo = {};
			ResourceInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM;
			ResourceInfo.descriptorSet = 0; // We assume that all bindings are in a single descriptor set.
			ResourceInfo.binding = Binding.VulkanBindingIdx;
			ResourceInfo.pNext = &ShapedModel->TensorInfosShaped[Binding.TensorId].VulkanDesc;
			DataGraphPipelineResourcesInfos.Add(ResourceInfo);
		}

		// Run the Vulkan resource creation functions on the RHI thread and wait for them to complete.
		FEvent* RenderThreadDoneEvent = FGenericPlatformProcess::GetSynchEventFromPool(true);
		ENQUEUE_RENDER_COMMAND(NNERuntimeRDGMLExtensionsForVulkanModel_CreateSegment)([&](FRHICommandListImmediate& RHICmdList) {
			RHICmdList.EnqueueLambda([&](FRHICommandListImmediate& RHICmdList) {
				VkDevice Device = GetIVulkanDynamicRHI()->RHIGetVkDevice();
				const VkAllocationCallbacks* Allocator = GetIVulkanDynamicRHI()->RHIGetVkAllocationCallbacks();

				// Shader module
				VkShaderModuleCreateInfo GraphShaderModuleCreateInfo = {};
				GraphShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				GraphShaderModuleCreateInfo.codeSize = ShapeInferenceResults.NewCode.Num() * sizeof(ShapeInferenceResults.NewCode[0]);
				GraphShaderModuleCreateInfo.pCode = ShapeInferenceResults.NewCode.GetData();
				VERIFYVULKANRESULT(vkCreateShaderModule_p(Device, &GraphShaderModuleCreateInfo, Allocator, &SegmentShaped.ShaderModule));

				// Data graph pipeline
				VkDataGraphPipelineShaderModuleCreateInfoARM DataGraphPipelineShaderModuleCreateInfo = {};
				DataGraphPipelineShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM;
				DataGraphPipelineShaderModuleCreateInfo.module = SegmentShaped.ShaderModule;
				DataGraphPipelineShaderModuleCreateInfo.pName = SegmentUnshaped.SPIRVEntryPoint;
				DataGraphPipelineShaderModuleCreateInfo.constantCount = DataGraphPipelineConstants.Num();
				DataGraphPipelineShaderModuleCreateInfo.pConstants = DataGraphPipelineConstants.GetData();

				VkDataGraphPipelineCreateInfoARM DataGraphPipelineCreateInfo = {};
				DataGraphPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM;
				DataGraphPipelineCreateInfo.layout = SegmentUnshaped.PipelineLayout;
				DataGraphPipelineCreateInfo.resourceInfoCount = DataGraphPipelineResourcesInfos.Num();
				DataGraphPipelineCreateInfo.pResourceInfos = DataGraphPipelineResourcesInfos.GetData();
				DataGraphPipelineCreateInfo.pNext = &DataGraphPipelineShaderModuleCreateInfo;

				VERIFYVULKANRESULT(vkCreateDataGraphPipelinesARM_p(Device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &DataGraphPipelineCreateInfo, Allocator, &SegmentShaped.Pipeline));
				});
			RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
			RenderThreadDoneEvent->Trigger();
			});
		RenderThreadDoneEvent->Wait();
		FGenericPlatformProcess::ReturnSynchEventToPool(RenderThreadDoneEvent);

		ShapedModel->SegmentsShaped.Add(MoveTemp(SegmentShaped));
	}

	// Fill in model output tensor shapes.
	ShapedModel->OutputTensorShapes.AddDefaulted(OutputSymbolicTensors.Num());
	for (int T = 0; T < TensorInfosUnshaped.Num(); ++T)
	{
		if (TensorInfosUnshaped[T].ModelOutputIdx != -1)
		{
			TArray<uint32_t> TensorShapeU32;
			Algo::Transform(ShapedModel->TensorInfosShaped[T].ShapeRawS64, TensorShapeU32, [](uint64_t x) { return x; });
			ShapedModel->OutputTensorShapes[TensorInfosUnshaped[T].ModelOutputIdx] = UE::NNE::FTensorShape::Make(TensorShapeU32);
		}
	}

	// Calculate NumBytes for each FTensorInfoShaped.
	for (FNNERuntimeRDGMLExtensionsForVulkanModelShaped::FTensorInfoShaped& TensorInfoShaped : ShapedModel->TensorInfosShaped)
	{
		size_t NumBytesPerElement = Private::GetNumBytesPerElement(TensorInfoShaped.VulkanDesc.format);
		if (NumBytesPerElement == 0)
		{
			UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Unsupported input/output/intermediate data type: %u"), TensorInfoShaped.VulkanDesc.format);
			return nullptr;
		}
		TensorInfoShaped.NumBytes = NumBytesPerElement * Algo::Accumulate(TensorInfoShaped.ShapeRawS64, 1, [](uint64_t Acc, uint64_t X) { return Acc * X; });
	}

	// Save in cache for future reuse.
	ShapedModels.Add(TArray<UE::NNE::FTensorShape>(ModelInputShapes), ShapedModel);
	return ShapedModel;
}

FNNERuntimeRDGMLExtensionsForVulkanModelShaped::~FNNERuntimeRDGMLExtensionsForVulkanModelShaped()
{
	// Destroy Vulkan resources on the RHI thread and wait for that to finish.
	FEvent* RenderThreadDoneEvent = FGenericPlatformProcess::GetSynchEventFromPool(true);
	ENQUEUE_RENDER_COMMAND(NNERuntimeRDGMLExtensionsForVulkanModel_DestroySegments)([&](FRHICommandListImmediate& RHICmdList) {
		RHICmdList.EnqueueLambda([&](FRHICommandListImmediate& RHICmdList) {
			VkDevice Device = GetIVulkanDynamicRHI()->RHIGetVkDevice();
			const VkAllocationCallbacks* Allocator = GetIVulkanDynamicRHI()->RHIGetVkAllocationCallbacks();

			for (FSegmentShaped& S : SegmentsShaped)
			{
				vkDestroyPipeline_p(Device, S.Pipeline, Allocator);
				vkDestroyShaderModule_p(Device, S.ShaderModule, Allocator);
			}

		});
		RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
		RenderThreadDoneEvent->Trigger();
	});

	RenderThreadDoneEvent->Wait();
	FGenericPlatformProcess::ReturnSynchEventToPool(RenderThreadDoneEvent);
}


FNNERuntimeRDGMLExtensionsForVulkanModelInstance::~FNNERuntimeRDGMLExtensionsForVulkanModelInstance()
{
	UnsetInputTensorShapes();

	// Destroy Vulkan resources on the RHI thread, and wait for that to finish.
	FEvent* RenderThreadDoneEvent = FGenericPlatformProcess::GetSynchEventFromPool(true);
	ENQUEUE_RENDER_COMMAND(NNERuntimeRDGMLExtensionsForVulkanModel_DestroySegments)([&](FRHICommandListImmediate& RHICmdList) {
		check(InFlightExecutions.IsEmpty());
		// Destroy resources
		RHICmdList.EnqueueLambda([&](FRHICommandListImmediate& RHICmdList) {
			VkDevice Device = GetIVulkanDynamicRHI()->RHIGetVkDevice();
			const VkAllocationCallbacks* Allocator = GetIVulkanDynamicRHI()->RHIGetVkAllocationCallbacks();

			vkDestroyDescriptorPool_p(Device, DescriptorPool, Allocator);
			DescriptorPool = VK_NULL_HANDLE;
		});

		RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
		RenderThreadDoneEvent->Trigger();
	});

	RenderThreadDoneEvent->Wait();
	FGenericPlatformProcess::ReturnSynchEventToPool(RenderThreadDoneEvent);
}

FNNERuntimeRDGMLExtensionsForVulkanModelInstance::ESetInputTensorShapesStatus FNNERuntimeRDGMLExtensionsForVulkanModelInstance::SetInputTensorShapes(TConstArrayView<UE::NNE::FTensorShape> InInputShapes)
{
	// This instance might already have been given a shape! In which case we might need to destroy the old set of things and recreate them.
	UnsetInputTensorShapes();
	
	// This is the first time that we could know the concrete shapes for all tensors, so we now need to run shape inference
	// through all the segments to determine all tensor shapes. This has to be done before we can create data graph pipelines etc.
	// We may already have performed shape inference on this model with the exact same input shapes, in which case we avoid doing it
	// again and instead share the same Shaped Model.
	ParentModelShaped = ParentModelUnshaped->FindOrCreateShapedModel(InInputShapes);
	if (ParentModelShaped == nullptr)
	{
		// There might have been an error doing shape inference, e.g. an invalid shape provided.
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Failed to infer shapes."));
		return ESetInputTensorShapesStatus::Fail;
	}

	// Now we can allocate inference-specific vulkan objects.
	TArray<FBufferRHIRef> PipelineSessionMemoryBuffers; // One per segment, stored temporarily until we can put them into FRDGPooledBuffers

	// Create vulkan resources for this instance, using the common resources from the parent model.
	// Run the Vulkan resource creation functions on the RHI thread and wait for them to complete.
	FEvent* RenderThreadDoneEvent = FGenericPlatformProcess::GetSynchEventFromPool(true);
	ENQUEUE_RENDER_COMMAND(NNERuntimeRDGMLExtensionsForVulkanModel_DestroySegments)([&](FRHICommandListImmediate& RHICmdList) {
		RHICmdList.EnqueueLambda([&](FRHICommandListImmediate& RHICmdList) {
			VkDevice Device = GetIVulkanDynamicRHI()->RHIGetVkDevice();
			const VkAllocationCallbacks* Allocator = GetIVulkanDynamicRHI()->RHIGetVkAllocationCallbacks();

			uint32 NumDescriptors = 0;
			for (int S = 0; S < ParentModelUnshaped->SegmentsUnshaped.Num(); ++S)
			{
				const FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped::FSegmentUnshaped& SegmentUnshaped = ParentModelUnshaped->SegmentsUnshaped[S];
				const FNNERuntimeRDGMLExtensionsForVulkanModelShaped::FSegmentShaped& SegmentShaped = ParentModelShaped->SegmentsShaped[S];
				FNNERuntimeRDGMLExtensionsForVulkanModelInstance::FSegmentInstance SegmentInstance;

				// Data graph pipeline session.
				VkDataGraphPipelineSessionCreateInfoARM DataGraphPipelineSessionCreateInfo = {};
				DataGraphPipelineSessionCreateInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM;
				DataGraphPipelineSessionCreateInfo.dataGraphPipeline = SegmentShaped.Pipeline;
				VERIFYVULKANRESULT(vkCreateDataGraphPipelineSessionARM_p(Device, &DataGraphPipelineSessionCreateInfo, Allocator, &SegmentInstance.DataGraphPipelineSession));

				// Find how much memory we need to allocate for the pipeline session.
				VkDataGraphPipelineSessionMemoryRequirementsInfoARM DataGraphPipelineSessionMemoryRequirementsInfo = {};
				DataGraphPipelineSessionMemoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM;
				DataGraphPipelineSessionMemoryRequirementsInfo.session = SegmentInstance.DataGraphPipelineSession;

				VkMemoryRequirements2 DataGraphPipelineSessionMemoryRequirements = {};
				DataGraphPipelineSessionMemoryRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
				vkGetDataGraphPipelineSessionMemoryRequirementsARM_p(Device, &DataGraphPipelineSessionMemoryRequirementsInfo, &DataGraphPipelineSessionMemoryRequirements);

				// There doesn't seem to be a publicly exposed way to allocate Vulkan memory,
				// so we allocate a buffer and then get its backing memory to use as our own.
				const FRHIBufferDesc BufferDesc = FRHIBufferDesc(DataGraphPipelineSessionMemoryRequirements.memoryRequirements.size, 0, EBufferUsageFlags::UnorderedAccess | EBufferUsageFlags::ByteAddressBuffer);
				FRHIResourceCreateInfo CreateInfo(TEXT("FNNERuntimeRDGMLExtensionsForVulkanModelInstance_PipelineSessionMemory"));
				FBufferRHIRef PipelineSessionMemoryBuffer = GetIVulkanDynamicRHI()->RHICreateBuffer(RHICmdList, BufferDesc, ERHIAccess::SRVCompute, CreateInfo);
				FVulkanRHIAllocationInfo AllocInfo = GetIVulkanDynamicRHI()->RHIGetAllocationInfo(PipelineSessionMemoryBuffer);
				PipelineSessionMemoryBuffers.Push(PipelineSessionMemoryBuffer);


				VkBindDataGraphPipelineSessionMemoryInfoARM BindDataGraphPipelineSessionMemoryInfo = {};
				BindDataGraphPipelineSessionMemoryInfo.sType = VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM;
				BindDataGraphPipelineSessionMemoryInfo.memory = AllocInfo.Handle;
				BindDataGraphPipelineSessionMemoryInfo.memoryOffset = AllocInfo.Offset;
				BindDataGraphPipelineSessionMemoryInfo.session = SegmentInstance.DataGraphPipelineSession;
				VERIFYVULKANRESULT(vkBindDataGraphPipelineSessionMemoryARM_p(Device, 1, &BindDataGraphPipelineSessionMemoryInfo));

				SegmentInstances.Add(SegmentInstance);

				// Sum up the total number of descriptors that we will need for all segments.
				NumDescriptors += SegmentUnshaped.Bindings.Num();
			}
		});

		RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);

		// Store pipeline session memory buffers into FRDGPooledBuffers for later use.
		for (int S = 0; S < SegmentInstances.Num(); ++S)
		{
			FRDGBufferDesc BufferDesc = FRDGBufferDesc::CreateByteAddressDesc(PipelineSessionMemoryBuffers[S]->GetSize());
			SegmentInstances[S].PipelineSessionMemoryPooledBuffer = new FRDGPooledBuffer(PipelineSessionMemoryBuffers[S], BufferDesc, 0, TEXT("FNNERuntimeRDGMLExtensionsForVulkanModelInstance_PipelineSessionMemory"));
		}

		RenderThreadDoneEvent->Trigger();
	});

	RenderThreadDoneEvent->Wait();
	FGenericPlatformProcess::ReturnSynchEventToPool(RenderThreadDoneEvent);

	return ESetInputTensorShapesStatus::Ok;
}

TConstArrayView<UE::NNE::FTensorDesc> FNNERuntimeRDGMLExtensionsForVulkanModelInstance::GetInputTensorDescs() const
{
	return ParentModelUnshaped->InputSymbolicTensors;
}

TConstArrayView<UE::NNE::FTensorDesc> FNNERuntimeRDGMLExtensionsForVulkanModelInstance::GetOutputTensorDescs() const
{
	return ParentModelUnshaped->OutputSymbolicTensors;
}

TConstArrayView<UE::NNE::FTensorShape> FNNERuntimeRDGMLExtensionsForVulkanModelInstance::GetInputTensorShapes() const
{
	// If SetInputTensorShapes hasn't been called yet then we won't know the input tensor shapes.
	return ParentModelShaped ? ParentModelShaped->InputTensorShapes : TConstArrayView<UE::NNE::FTensorShape>{};
}

TConstArrayView<UE::NNE::FTensorShape> FNNERuntimeRDGMLExtensionsForVulkanModelInstance::GetOutputTensorShapes() const
{
	// If SetInputTensorShapes hasn't been called yet then we won't know the output tensor shapes.
	return ParentModelShaped ? ParentModelShaped->OutputTensorShapes : TConstArrayView<UE::NNE::FTensorShape>{};
}

BEGIN_SHADER_PARAMETER_STRUCT(FRDGPassParameters, )
	RDG_BUFFER_ACCESS_ARRAY(TensorBuffers)
	RDG_BUFFER_ACCESS_ARRAY(PipelineSessionMemoryBuffers)
END_SHADER_PARAMETER_STRUCT()

FNNERuntimeRDGMLExtensionsForVulkanModelInstance::EEnqueueRDGStatus FNNERuntimeRDGMLExtensionsForVulkanModelInstance::EnqueueRDG(FRDGBuilder& RDGBuilder, TConstArrayView<UE::NNE::FTensorBindingRDG> ModelInputs,
	TConstArrayView<UE::NNE::FTensorBindingRDG> ModelOutputs)
{
	check(IsInRenderingThread());

	// Check that shape inference has been performed (i.e. SetInputTensorShapes was called).
	if (!ParentModelShaped)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Please call SetInputTensorShapes before calling EnqueueRDG"));
		return EEnqueueRDGStatus::Fail;
	}

	// Validate that the number of inputs/outputs is as expected. 
	// We don't have too much detail about the buffers themselves so can't validate formats and shapes, but we can at least validate the total byte size
	// (which we do in the below loop).
	if (ModelInputs.Num() != ParentModelShaped->InputTensorShapes.Num() || ModelOutputs.Num() != ParentModelShaped->OutputTensorShapes.Num())
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Incorrect number of inputs or outputs"));
		return EEnqueueRDGStatus::Fail;
	}


	// Make an array of all the RDG buffers we need - one for each input/output/intermediate tensor, in the same order as our TensorInfos.
	FRDGPassParameters* RDGPassParams = RDGBuilder.AllocParameters<FRDGPassParameters>();
	for (int T = 0; T < ParentModelUnshaped->TensorInfosUnshaped.Num(); ++T)
	{
		const FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped::FTensorInfoUnshaped& TensorInfoUnshaped = ParentModelUnshaped->TensorInfosUnshaped[T];
		const FNNERuntimeRDGMLExtensionsForVulkanModelShaped::FTensorInfoShaped& TensorInfoShaped = ParentModelShaped->TensorInfosShaped[T];
		if (TensorInfoUnshaped.IsIntermediate())
		{
			// We use RDG for intermediate tensors so that it can re-use memory etc. rather than allocating them up-front.
			FRDGBufferDesc BufferDesc = FRDGBufferDesc::CreateByteAddressDesc(TensorInfoShaped.NumBytes);
			FRDGBufferRef Buffer = RDGBuilder.CreateBuffer(BufferDesc, TEXT("FNNERuntimeRDGMLExtensionsForVulkanModelInstance_Intermediate"), ERDGBufferFlags::None);
			RDGPassParams->TensorBuffers.Emplace(Buffer, ERHIAccess::UAVCompute);
		}
		else if (TensorInfoUnshaped.ModelInputIdx >= 0)
		{
			FRDGBufferRef RDGBuffer = ModelInputs[TensorInfoUnshaped.ModelInputIdx].Buffer;
			RDGPassParams->TensorBuffers.Emplace(RDGBuffer, ERHIAccess::SRVCompute);
			if (RDGBuffer->GetSize() < TensorInfoShaped.NumBytes)
			{
				UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Input buffer is too small"));
				return EEnqueueRDGStatus::Fail;
			}
		}
		else if (TensorInfoUnshaped.ModelOutputIdx >= 0)
		{
			FRDGBufferRef RDGBuffer = ModelOutputs[TensorInfoUnshaped.ModelOutputIdx].Buffer;
			RDGPassParams->TensorBuffers.Emplace(RDGBuffer, ERHIAccess::UAVCompute);
			if (RDGBuffer->GetSize() < TensorInfoShaped.NumBytes)
			{
				UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Output buffer is too small"));
				return EEnqueueRDGStatus::Fail;
			}
		}
	}
	// Also include all the buffers we created to hold the pipeline session memory, so that these are tracked correctly.
	for (const FSegmentInstance& S : SegmentInstances)
	{
		RDGPassParams->PipelineSessionMemoryBuffers.Emplace(RDGBuilder.RegisterExternalBuffer(S.PipelineSessionMemoryPooledBuffer), ERHIAccess::UAVCompute);
	}

	RDGBuilder.AddPass(
		RDG_EVENT_NAME("FNNERuntimeRDGMLExtensionsForVulkanModelInstance_SegmentInstance"),
		RDGPassParams,
		ERDGPassFlags::Compute,
		[RDGPassParams, &InFlightExecutions = InFlightExecutions, this, ParentModelShaped = this->ParentModelShaped.Get(), ParentModelUnshaped = this->ParentModelUnshaped.Get(),
		 DescriptorPool = DescriptorPool, &SegmentInstances = this->SegmentInstances](FRHICommandListImmediate& RHICmdList)
		{
			// Get the RHI buffers from the RDG buffers.
			TArray<FRHIBuffer*> RHIBuffers;
			for (FRDGBuffer* RDGBuffer : RDGPassParams->TensorBuffers)
			{
				RDGBuffer->MarkResourceAsUsed();
				RHIBuffers.Add(RDGBuffer->GetRHI());
			}

			// Mark pipeline session memory buffers as used, to ensure they are tracked properly.
			for (FRDGBuffer* PipelineSessionMemoryBuffer : RDGPassParams->PipelineSessionMemoryBuffers)
			{
				PipelineSessionMemoryBuffer->MarkResourceAsUsed();
			}

			// Clean up any finished executions and wait until we have a free one 
			// (otherwise we would try to allocate too many descriptor sets).
			CleanupFinishedExecutions(RHICmdList);
			while (InFlightExecutions.Num() >= MAX_CONCURRENT_EXECUTIONS_PER_INSTANCE)
			{
				// We need to flush the RHI thread otherwise we might deadlock.
				RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
				CleanupFinishedExecutions(RHICmdList);
			}

			// This is a new execution.
			InFlightExecutions.PushLast(FExecution{});
			FExecution& Execution = InFlightExecutions.Last();

			// Create resources and submit the graph inference on the RHI thread.
			RHICmdList.EnqueueLambda([RHIBuffers = MoveTemp(RHIBuffers), &Execution, ParentModelShaped, ParentModelUnshaped, DescriptorPool, &SegmentInstances](FRHICommandListImmediate& RHICmdList) {
				VkDevice Device = GetIVulkanDynamicRHI()->RHIGetVkDevice();
				const VkAllocationCallbacks* Allocator = GetIVulkanDynamicRHI()->RHIGetVkAllocationCallbacks();

				// Create resources for this execution.
				// VkTensors and VkTensorViews for all inputs, outputs and intermediates (between segments).
				Execution.VulkanTensors.Reserve(RHIBuffers.Num());
				Execution.VulkanTensorViews.Reserve(RHIBuffers.Num());
				for (int32 TensorId = 0; TensorId < RHIBuffers.Num(); ++TensorId)
				{
					VkTensorCreateInfoARM TensorCreateInfo = {};
					TensorCreateInfo.sType = VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM;
					TensorCreateInfo.pDescription = &ParentModelShaped->TensorInfosShaped[TensorId].VulkanDesc;
					VkTensorARM VulkanTensor;
					VERIFYVULKANRESULT(vkCreateTensorARM_p(Device, &TensorCreateInfo, Allocator, &VulkanTensor));
					Execution.VulkanTensors.Add(VulkanTensor);

					VkBindTensorMemoryInfoARM BindTensorMemoryInfo = {};
					BindTensorMemoryInfo.sType = VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM;
					BindTensorMemoryInfo.tensor = VulkanTensor;

					FRHIBuffer* RHIBuffer = RHIBuffers[TensorId];
					const FVulkanRHIAllocationInfo& Allocation = GetIVulkanDynamicRHI()->RHIGetAllocationInfo(RHIBuffer);
					BindTensorMemoryInfo.memory = Allocation.Handle;
					BindTensorMemoryInfo.memoryOffset = Allocation.Offset;

					VERIFYVULKANRESULT(vkBindTensorMemoryARM_p(Device, 1, &BindTensorMemoryInfo));

					VkTensorViewCreateInfoARM TensorViewCreateInfo = {};
					TensorViewCreateInfo.sType = VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM;
					TensorViewCreateInfo.format = TensorCreateInfo.pDescription->format;
					TensorViewCreateInfo.tensor = VulkanTensor;
					VkTensorViewARM VulkanTensorView;
					VERIFYVULKANRESULT(vkCreateTensorViewARM_p(Device, &TensorViewCreateInfo, Allocator, &VulkanTensorView));
					Execution.VulkanTensorViews.Add(VulkanTensorView);
				}

				// Descriptor sets for each segment.
				Execution.DescriptorSets.AddZeroed(ParentModelShaped->SegmentsShaped.Num());
				for (int S = 0; S < ParentModelShaped->SegmentsShaped.Num(); ++S)
				{
					// Allocate a new descriptor set.
					VkDescriptorSetAllocateInfo AllocInfo = {};
					AllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
					AllocInfo.descriptorPool = DescriptorPool;
					AllocInfo.descriptorSetCount = 1;
					AllocInfo.pSetLayouts = &ParentModelUnshaped->SegmentsUnshaped[S].DescriptorSetLayout;
					VkDescriptorSet& DescriptorSet = Execution.DescriptorSets[S];
					VERIFYVULKANRESULT(vkAllocateDescriptorSets_p(Device, &AllocInfo, &DescriptorSet));

					// Update descriptor sets to bind the input/output buffers for this segment
					const TArray<FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped::FSegmentUnshaped::FBinding>& Bindings = ParentModelUnshaped->SegmentsUnshaped[S].Bindings;
					TArray<VkWriteDescriptorSetTensorARM> TensorInfos;
					TArray<VkWriteDescriptorSet> DescriptorSetWrites;
					TensorInfos.Reserve(Bindings.Num());
					DescriptorSetWrites.Reserve(Bindings.Num());
					for (int B = 0; B < Bindings.Num(); ++B)
					{
						VkWriteDescriptorSetTensorARM& TensorInfo = TensorInfos.AddZeroed_GetRef();
						TensorInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM;
						TensorInfo.tensorViewCount = 1;
						TensorInfo.pTensorViews = &Execution.VulkanTensorViews[Bindings[B].TensorId];

						VkWriteDescriptorSet DescriptorSetWrite = {};
						DescriptorSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						DescriptorSetWrite.pNext = &TensorInfo;
						DescriptorSetWrite.descriptorCount = 1;
						DescriptorSetWrite.dstSet = DescriptorSet;
						DescriptorSetWrite.dstBinding = Bindings[B].VulkanBindingIdx;
						DescriptorSetWrite.descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;

						DescriptorSetWrites.Add(DescriptorSetWrite);
					}

					vkUpdateDescriptorSets_p(Device, DescriptorSetWrites.Num(), DescriptorSetWrites.GetData(), 0, NULL);

					// Finally we can add the command to run the graph.
					VkCommandBuffer CommandBuffer = GetIVulkanDynamicRHI()->RHIGetActiveVkCommandBuffer();
					vkCmdBindDescriptorSets_p(CommandBuffer, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, ParentModelUnshaped->SegmentsUnshaped[S].PipelineLayout, 0, 1, &DescriptorSet, 0, NULL);
					vkCmdBindPipeline_p(CommandBuffer, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, ParentModelShaped->SegmentsShaped[S].Pipeline);
					vkCmdDispatchDataGraphARM_p(CommandBuffer, SegmentInstances[S].DataGraphPipelineSession, NULL);

					// As we've messed about with the Vulkan state, tell the RHI to reset it.
					GetIVulkanDynamicRHI()->RHIFinishExternalComputeWork(CommandBuffer);
				}
			});

			// Create and store a GPU fence so that we can tell when this execution has finished.
			Execution.GPUFence = RHICreateGPUFence("FNNERuntimeRDGMLExtensionsForVulkanModelInstance_Execution");
			RHICmdList.WriteGPUFence(Execution.GPUFence);
		}
	);

	return EEnqueueRDGStatus::Ok;
}

void FNNERuntimeRDGMLExtensionsForVulkanModelInstance::UnsetInputTensorShapes()
{
	// Destroy Vulkan resources on the RHI thread, and wait for that to finish.
	FEvent* RenderThreadDoneEvent = FGenericPlatformProcess::GetSynchEventFromPool(true);
	ENQUEUE_RENDER_COMMAND(NNERuntimeRDGMLExtensionsForVulkanModel_DestroySegments)([&](FRHICommandListImmediate& RHICmdList) {
		// Wait for any outstanding executions to finish.
		while (!InFlightExecutions.IsEmpty())
		{
			FPlatformProcess::Sleep(0.0f);
			// We need to flush the RHI thread otherwise we might deadlock.
			RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
			CleanupFinishedExecutions(RHICmdList);
		}

		// Destroy resources
		RHICmdList.EnqueueLambda([&](FRHICommandListImmediate& RHICmdList) {
			VkDevice Device = GetIVulkanDynamicRHI()->RHIGetVkDevice();
			const VkAllocationCallbacks* Allocator = GetIVulkanDynamicRHI()->RHIGetVkAllocationCallbacks();

			for (FSegmentInstance& S : SegmentInstances)
			{
				vkDestroyDataGraphPipelineSessionARM_p(Device, S.DataGraphPipelineSession, Allocator);
			}
			SegmentInstances.Empty(); // Destroy the textures on the render thread (rather than letting the default destructor run on the game thread).
		});

		RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
		RenderThreadDoneEvent->Trigger();
	});

	RenderThreadDoneEvent->Wait();
	FGenericPlatformProcess::ReturnSynchEventToPool(RenderThreadDoneEvent);
	
	// Note that this model instance object may still be re-used afterwards if it is given new tensor shapes,
	// so restore everything to sensible defaults.
	ParentModelShaped.Reset();
}

void FNNERuntimeRDGMLExtensionsForVulkanModelInstance::CleanupFinishedExecutions(FRHICommandListImmediate& RHICmdList)
{
	check(IsInRenderingThread());

	while (!InFlightExecutions.IsEmpty() && InFlightExecutions.First().GPUFence->Poll())
	{
		// Clean up and remove this execution on the RHI thread.
		FExecution& Execution = InFlightExecutions.First();

		RHICmdList.EnqueueLambda([Execution = MoveTemp(Execution), DescriptorPool = DescriptorPool](FRHICommandListImmediate& RHICmdList) {
			VkDevice Device = GetIVulkanDynamicRHI()->RHIGetVkDevice();
			const VkAllocationCallbacks* Allocator = GetIVulkanDynamicRHI()->RHIGetVkAllocationCallbacks();
			
			VERIFYVULKANRESULT(vkFreeDescriptorSets_p(Device, DescriptorPool, Execution.DescriptorSets.Num(), Execution.DescriptorSets.GetData()));
			for (VkTensorViewARM TensorView : Execution.VulkanTensorViews)
			{
				vkDestroyTensorViewARM_p(Device, TensorView, Allocator);
			}
			for (VkTensorARM Tensor : Execution.VulkanTensors)
			{
				vkDestroyTensorARM_p(Device, Tensor, Allocator);
			}
		});

		InFlightExecutions.PopFirst();
	}
}
