// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "NNERuntime.h"
#include "NNERuntimeRDG.h"

#include "NNERuntimeRDGMLExtensionsForVulkan.generated.h"

/// The NNE runtime for ML Extensions for Vulkan®. A single instance of this class is created and registered with the NNE runtime.
UCLASS()
class UNNERuntimeRDGMLExtensionsForVulkan : public UObject, public INNERuntime, public INNERuntimeRDG
{
	GENERATED_BODY()

public:
	// ID and version to identify the compiled model data.
	static FGuid ModelDataGUID;
	static int32 ModelDataVersion;

	bool SupportsInference;

	UNNERuntimeRDGMLExtensionsForVulkan() {};
	virtual ~UNNERuntimeRDGMLExtensionsForVulkan() {}

	virtual FString GetRuntimeName() const override;

	virtual INNERuntime::ECanCreateModelDataStatus CanCreateModelData(const FString& FileType, TConstArrayView64<uint8> FileData,
		const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId,
		const ITargetPlatform* TargetPlatform) const override;
	virtual TSharedPtr<UE::NNE::FSharedModelData> CreateModelData(const FString& FileType, TConstArrayView64<uint8> FileData,
		const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId,
		const ITargetPlatform* TargetPlatform) override;

	virtual FString GetModelDataIdentifier(const FString& FileType, TConstArrayView64<uint8> FileData,
		const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId,
		const ITargetPlatform* TargetPlatform) const override;

	virtual INNERuntimeRDG::ECanCreateModelRDGStatus CanCreateModelRDG(TObjectPtr<UNNEModelData> ModelData) const override;
	virtual TSharedPtr<UE::NNE::IModelRDG> CreateModelRDG(TObjectPtr<UNNEModelData> ModelData) override;
};
