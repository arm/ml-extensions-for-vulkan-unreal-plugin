// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "Templates/SharedPointer.h"
#include "NNEModelData.h"
#include "NNERuntimeRDG.h"
#include "IVulkanDynamicRHI.h"
#include "Containers/Deque.h"
#include "RenderGraphResources.h"

// There are three model classes in this file so that data can be shared between different instances of the same model. There is a one-to-many
// relationship between these: One 'unshaped model' can be used by many 'shaped models' and one 'shaped model' can be used by many 'model instances'.
//  1. FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped - this corresponds to NNE's IModelRDG and contains a potentially-unshaped model description.
//         This can't have Vulkan pipelines created for it, because it might not have a concrete shape, but can store information
//         about the VGF and bindings etc. This is shared across all instances that use the model, even if they have different tensor shapes.
//			Note that, depending on the model, the VGF might actually have all concrete shapes, but we still call this 'unshaped' for consistency.
//  2. FNNERuntimeRDGMLExtensionsForVulkanModelShaped - this doesn't have a corresponding NNE type, but contains a fully shaped version of
//         the model. This can be shared between different instances that use the same tensor shapes. As it doesn't have an NNE equivalent,
//         these are stored in a cache inside the unshaped model, so that if multiple model instances use the same tensor shapes, they
//			can re-use the same shaped model.
//  3. FNNERuntimeRDGMLExtensionsForVulkanModelInstance - this corresponds to NNE's IModelInstanceRDG and contains a pipeline session so that it can
//         be used to run inferences.

// The unshaped model class parses the VGF file and creates some of the Vulkan resources which can be shared amongst
// any shaped models using this model. For example, shader modules are not created here as they depend on shape
// information that might not be present, and intermediate buffers are not allocated here as they would
// need to be unique for each inference, but constant buffers that are the same for every inference can be created here.
class FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped : public UE::NNE::IModelRDG, public TSharedFromThis<FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped>
{
public:
	static TSharedPtr<FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped> Create(const TSharedPtr<UE::NNE::FSharedModelData>& InModelData);

	virtual ~FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped();

	virtual TSharedPtr<UE::NNE::IModelInstanceRDG> CreateModelInstanceRDG() override;

private:
	FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped();

	// If a shaped model already exists with the given input shapes, return it. If not, create a new one.
	TSharedPtr<class FNNERuntimeRDGMLExtensionsForVulkanModelShaped> FindOrCreateShapedModel(TConstArrayView<UE::NNE::FTensorShape> ModelInputShapes);

	// It's important that we keep a shared pointer to model data, as this contains the VGF binary (with constants and SPIR-V code)
	// which we need to use later on (after the Create function has returned). NNE does not guarantee that the model data
	// will be kept around after this point, so we have to do it here.
	TSharedPtr<UE::NNE::FSharedModelData> SharedModelData;

	// The VGF format describes a connected graph of 'segments', where each segment is either a Compute shader
	// or an ML Extensions for Vulkan Graph. This struct contains the information about a segment that we need to run it,
	// but only common information that is shared between all shaped models (see also FSegmentInstance and FSegmentShaped).
	struct FSegmentUnshaped
	{
		// Information about an input or output of a segment.
		struct FBinding
		{
			uint32 VulkanBindingIdx; // The binding number in the Vulkan descriptor set for this segment.
			// Lookup into TensorInfos for details about this tensor. This is how we match up the outputs of one segment
			// with the inputs of another.
			uint32 TensorId;
			// Is this binding for an input or output of the segment.
			enum class EBindingKind
			{
				Input,
				Output
			} BindingKind;
		};

		// Information about a constant in this segment.
		struct FConstantInfo
		{
			// This contains pointers to constant data embedded in the VGF. The underlying data is kept alive by the SharedModelData shared ptr.
			VkDataGraphPipelineConstantARM DataGraphPipelineConstant;
			VkTensorDescriptionARM TensorDescription;
			TArray<int64_t> TensorDimensions; // Storage for TensorDescription.pDimensions.
		};

		FString Name; // Only for debugging, no effect on behaviour.
		VkDescriptorSetLayout DescriptorSetLayout;
		VkPipelineLayout PipelineLayout;
		TArray<FBinding> Bindings; // Inputs and outputs for this segment.
		TConstArrayView<uint32_t> SPIRVCode; // This is a view of the SPIR-V code embedded in the VGF. The underlying data is kept alive by the SharedModelData shared ptr.
		const char* SPIRVEntryPoint; // This is a raw pointer to a string embedded in the VGF. This data is kept alive by the SharedModelData shared ptr.
		// Information about constants in this segment. As we don't create the pipeline until shape has been inferred, we need to keep this around.
		TArray<FConstantInfo> ConstantInfos;
	};

	TArray<FSegmentUnshaped> SegmentsUnshaped;

	// Details about the whole model's inputs and outputs, which we pass down to the model instance for access from its public API.
	TArray<UE::NNE::FTensorDesc>  InputSymbolicTensors;
	TArray<UE::NNE::FTensorDesc>  OutputSymbolicTensors;

	// Description of an input, output or intermediate (between segments) tensor.
	struct FTensorInfoUnshaped
	{
		int32 ModelInputIdx = -1; // If this is a model input tensor, this says which input number it is. -1 means not an input.
		int32 ModelOutputIdx = -1; // If this is a model output tensor, this says which output number it is. -1 means not an output.
		VkTensorDescriptionARM VulkanDesc; // Note that the shape (pDimensions) in here will be nullptr, as it hasn't been shaped yet. It does however have format etc.

		bool IsIntermediate() const { return ModelInputIdx == -1 && ModelOutputIdx == -1; }
	};

	// Descriptions of input, output and intermediate (between segment) tensors. 
	// We don't need to store anything about other tensor types (e.g. constants or within-segment intermediates) 
	// as these are hidden inside the data graph.
	// The index into this array is the 'TensorId' which is a concept we create on top of VGF so that we have contiguous IDs.
	TArray<FTensorInfoUnshaped> TensorInfosUnshaped;

	// Cache for all of the shaped models that have been created for this unshaped model.
	// Multiple model instances can use the same shaped model and when the last instance dies this shaped model
	// will be freed. We deliberately use weak ptr so that this cache doesn't keep the shaped model alive indefinitely.
	TMap<TArray<UE::NNE::FTensorShape>, TWeakPtr<FNNERuntimeRDGMLExtensionsForVulkanModelShaped>> ShapedModels;

	friend class FNNERuntimeRDGMLExtensionsForVulkanModelInstance;
};

// The shaped model class builds upon an unshaped model and has concrete shapes for every tensor.
// This allocates Vulkan resources for shader modules and data graph pipelines.
// Common resources shared between different shaped models are simply referenced from the 'parent' unshaped model.
class FNNERuntimeRDGMLExtensionsForVulkanModelShaped
{
public:
	~FNNERuntimeRDGMLExtensionsForVulkanModelShaped();

private:
	// Reference to common data shared between all shaped models which are based on the same unshaped model.
	// Importantly the smart pointer also prevents the common data from being destroyed whilst we are still using it.
	TSharedPtr<FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped> ParentModelUnshaped;

	// Details about the whole model's inputs and outputs, which we pass down to the model instance for access from its public API.
	TArray<UE::NNE::FTensorShape> InputTensorShapes;
	TArray<UE::NNE::FTensorShape> OutputTensorShapes;

	// Information needed about a segment that is unique for each shaped model, but shared between model instances.
	// (as opposed to the information in the parent unshaped model's FSegmentUnshaped, which is shared).
	struct FSegmentShaped
	{
		VkShaderModule ShaderModule;
		VkPipeline Pipeline;
	};

	TArray<FSegmentShaped> SegmentsShaped;

	// Description of an input, output or intermediate (between segments) tensor, with concrete shape specified
	// (FTensorInfoUnshaped might not have a concrete shape).
	struct FTensorInfoShaped
	{
		VkTensorDescriptionARM VulkanDesc; // Note that the shape in here might have some -1s for dimensions which haven't been specified
		TArray<int64_t> ShapeRawS64; // Storage for the pDimensions pointer in VkTensorDescriptionARM.
		uint32 NumBytes = 0;
	};

	// The index into this array is the same 'TensorId' concept from the unshaped model.
	TArray<FTensorInfoShaped> TensorInfosShaped;

	friend class FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped;
	friend class FNNERuntimeRDGMLExtensionsForVulkanModelInstance;
};

// The model instance class builds upon a shaped model and adds the Vulkan resources to run an inference for a model,
// including a pipeline session object.
// Common resources shared between different inferences are simply referenced from the 'parent' shaped model.
// The lifecycle of this class is a bit weird/awkward, because a lot of the resources it manages can't be created
// until the tensor shapes are known, i.e. after SetInputTensorShapes is called. SetInputTensorShapes can also
// be called multiple times during its lifetime, so these resources may need to be recreated multiple times.
class FNNERuntimeRDGMLExtensionsForVulkanModelInstance : public UE::NNE::IModelInstanceRDG
{
public:
	FNNERuntimeRDGMLExtensionsForVulkanModelInstance() {}
	virtual ~FNNERuntimeRDGMLExtensionsForVulkanModelInstance();

	virtual TConstArrayView<UE::NNE::FTensorDesc> GetInputTensorDescs() const override;
	virtual TConstArrayView<UE::NNE::FTensorDesc> GetOutputTensorDescs() const override;
	virtual TConstArrayView<UE::NNE::FTensorShape> GetInputTensorShapes() const override;
	virtual TConstArrayView<UE::NNE::FTensorShape> GetOutputTensorShapes() const override;
	virtual ESetInputTensorShapesStatus SetInputTensorShapes(TConstArrayView<UE::NNE::FTensorShape> InInputShapes) override;

	virtual ESetInputTensorShapesStatus EnqueueRDG(FRDGBuilder& RDGBuilder, TConstArrayView<UE::NNE::FTensorBindingRDG> Inputs,
		TConstArrayView<UE::NNE::FTensorBindingRDG> Outputs) override;
private:
	void UnsetInputTensorShapes(); // Destroys all resources created as a result of SetInputTensorShapes.
	void CleanupFinishedExecutions(FRHICommandListImmediate& RHICmdList);

	// Reference to common data (shared between all model instances of this model).
	// Importantly the smart pointer also prevents the common data from being destroyed whilst we are still using it.
	TSharedPtr<FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped> ParentModelUnshaped;
	// Reference to common data (shared between all model instances of the same shaped model).
	// Importantly the smart pointer also prevents the common data from being destroyed whilst we are still using it.
	TSharedPtr<FNNERuntimeRDGMLExtensionsForVulkanModelShaped> ParentModelShaped;

	// Information needed about a segment that is unique for each model instance
	// (as opposed to the information in the parent model's FSegmentShaped, which is shared).
	struct FSegmentInstance
	{
		VkDataGraphPipelineSessionARM DataGraphPipelineSession;
		// Buffer object which owns the memory that we use for the Graph Pipeline Session.
		// (This is never actually used as a buffer!)
		TRefCountPtr<FRDGPooledBuffer> PipelineSessionMemoryPooledBuffer;
	};

	// An FSegmentInstance for each Segment in the model.
	TArray<FSegmentInstance> SegmentInstances;

	// Pool that we use to allocate all the descriptor sets (one per segment) from.
	VkDescriptorPool DescriptorPool;

	// Resources being used by a single execution of the model. These can't be destroyed/modified/re-used
	// until after that execution has finished, which might be after we have queued up the next one.
	struct FExecution
	{
		TArray<VkDescriptorSet> DescriptorSets; // One for each segment
		TArray<VkTensorARM> VulkanTensors; // One for each tensor in TensorInfos.
		TArray<VkTensorViewARM> VulkanTensorViews; // One for each tensor in TensorInfos.
		FGPUFenceRHIRef GPUFence; // Tells us when the GPU has finished with this execution, so that we can free the resources in here.
	};

	// There can be multiple executions of this model instance in-flight at the same time as the render thread can be queuing
	// up commands for the next frame whilst the GPU is still rendering the previous one.
	// This array should only be modified by the rendering thread to avoid synchronisation problems.
	TDeque<FExecution> InFlightExecutions;

	friend class FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped;
};
