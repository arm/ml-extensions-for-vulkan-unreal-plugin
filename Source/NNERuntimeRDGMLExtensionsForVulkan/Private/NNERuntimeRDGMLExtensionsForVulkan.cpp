// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "NNERuntimeRDGMLExtensionsForVulkan.h"
#include "NNERuntimeRDGMLExtensionsForVulkanModule.h"
#include "NNEModelData.h"
#include "Serialization/MemoryWriter.h"
#include "Misc/FileHelper.h"
#include "NNERuntimeRDGMLExtensionsForVulkanModel.h"

using namespace UE::NNE;

FGuid UNNERuntimeRDGMLExtensionsForVulkan::ModelDataGUID = FGuid((int32)'N', (int32)'A', (int32)'M', (int32)'V');
int32 UNNERuntimeRDGMLExtensionsForVulkan::ModelDataVersion = 1;

FString UNNERuntimeRDGMLExtensionsForVulkan::GetRuntimeName() const
{
	return TEXT("NNERuntimeRDGMLExtensionsForVulkan");
}

INNERuntime::ECanCreateModelDataStatus UNNERuntimeRDGMLExtensionsForVulkan::CanCreateModelData(const FString& FileType, TConstArrayView64<uint8> FileData,
	const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform) const
{
	if (!FileType.Equals("vgf", ESearchCase::IgnoreCase))
	{
		// No need to log an error here, as this will be reported by the layer above. In fact it could fail the cooking commandlet
		// if we reported an error here and there were model assets with unsupported file types that Unreal attempts to cook
		// with this runtime.
		return ECanCreateModelDataStatus::FailFileIdNotSupported;
	}

	return ECanCreateModelDataStatus::Ok;
}

TSharedPtr<FSharedModelData> UNNERuntimeRDGMLExtensionsForVulkan::CreateModelData(const FString& FileType, TConstArrayView64<uint8> FileData,
	const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform)
{
	if (CanCreateModelData(FileType, FileData, AdditionalFileData, FileId, TargetPlatform) != ECanCreateModelDataStatus::Ok)
	{
		// Error will have been logged by CanCreateModelData.
		return TSharedPtr<FSharedModelData>();
	}

	TArray<uint8> VGFBuffer;

	if (FileType.Equals("vgf", ESearchCase::IgnoreCase))
	{
		// Nothing to do, use VGF data as is.
		VGFBuffer = FileData;
	}
	else
	{
		// Shouldn't get here, but just in case.
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Unsupported file type."));
		return TSharedPtr<FSharedModelData>();
	}

	TArray<uint8> ModelData;
	FMemoryWriter Writer(ModelData);
	// Prepend GUID and version so that we can later detect corrupt or old versions.
	Writer << ModelDataGUID;
	Writer << ModelDataVersion;
	Writer.Serialize(VGFBuffer.GetData(), VGFBuffer.Num());

	return MakeShared<FSharedModelData>(MakeSharedBufferFromArray(MoveTemp(ModelData)), 0);
}

FString UNNERuntimeRDGMLExtensionsForVulkan::GetModelDataIdentifier(const FString& FileType, TConstArrayView64<uint8> FileData,
	const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform) const
{
	return FileId.ToString(EGuidFormats::Digits) + "-" + ModelDataGUID.ToString(EGuidFormats::Digits) + "-" + FString::FromInt(UNNERuntimeRDGMLExtensionsForVulkan::ModelDataVersion);
}

INNERuntimeRDG::ECanCreateModelRDGStatus UNNERuntimeRDGMLExtensionsForVulkan::CanCreateModelRDG(TObjectPtr<UNNEModelData> ModelData) const
{
	check(ModelData != nullptr);

	if (!SupportsInference)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Missing support for inference - see earlier log messages from NNERuntimeRDGMLExtensionsForVulkan."))
		return ECanCreateModelRDGStatus::Fail;
	}

	// Check that the UNNEModelData contains valid data for this NNE runtime and that it's the current version
	const TSharedPtr<FSharedModelData> ModelDataForThisRuntime = ModelData->GetModelData(GetRuntimeName());
	if (!ModelDataForThisRuntime.IsValid())
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("UNNEModelData is missing data for this runtime."))
		return ECanCreateModelRDGStatus::Fail;
	}

	TConstArrayView64<uint8> Data = ModelDataForThisRuntime->GetView();
	if (Data.Num() <= sizeof(ModelDataGUID) + sizeof(ModelDataVersion))
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("UNNEModelData model data for this runtime is too small."))
		return ECanCreateModelRDGStatus::Fail;
	}

	// Validate the GUID which should be the first thing in the data
	if (FGenericPlatformMemory::Memcmp(&Data[0], &ModelDataGUID, sizeof(ModelDataGUID)) != 0)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("UNNEModelData model data for this runtime has incorrect GUID."))
		return ECanCreateModelRDGStatus::Fail;
	}

	// Validate the version number which should be immediately after the GUID
	if (FGenericPlatformMemory::Memcmp(&Data[sizeof(ModelDataGUID)], &ModelDataVersion, sizeof(ModelDataVersion)) != 0)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("UNNEModelData model data for this runtime has incorrect version."))
		return ECanCreateModelRDGStatus::Fail;
	}

	return ECanCreateModelRDGStatus::Ok;
}

TSharedPtr<IModelRDG> UNNERuntimeRDGMLExtensionsForVulkan::CreateModelRDG(TObjectPtr<UNNEModelData> ModelData)
{
	check(ModelData != nullptr);

	if (CanCreateModelRDG(ModelData) != ECanCreateModelRDGStatus::Ok)
	{
		// Error will have been logged by CanCreateModelRDG
		return TSharedPtr<IModelRDG>();
	}

	const TSharedPtr<FSharedModelData> ModelDataForThisRuntime = ModelData->GetModelData(GetRuntimeName());
	check(ModelData != nullptr); // Already validated by CanCreateModelRDG

	return FNNERuntimeRDGMLExtensionsForVulkanModelUnshaped::Create(ModelDataForThisRuntime);
}
