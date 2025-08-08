// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#include "Factories/Factory.h"

#include "NNERuntimeRDGMLExtensionsForVulkanModelDataFactory.generated.h"

/// Simple asset factory which takes .vgf files and creates a UNNEModelData asset for them.
/// This is pretty much identical to the vanilla UNNEModelDataFactory, but declares support for .vgf files instead of .onnx.
/// It also gives us the option of adding custom import settings that are specific for our runtime, for example shape overrides
/// (see the UNNERuntimeIREEModelDataFactory class for an example of this.)
UCLASS()
class UNNERuntimeRDGMLExtensionsForVulkanModelDataFactory : public UFactory
{
	GENERATED_BODY()

public:
	UNNERuntimeRDGMLExtensionsForVulkanModelDataFactory(const FObjectInitializer& ObjectInitializer);

public:
	//~ Begin UFactory Interface
	virtual UObject* FactoryCreateBinary(UClass* Class, UObject* InParent, FName Name, EObjectFlags Flags, UObject* Context, const TCHAR* Type, const uint8*& Buffer, const uint8* BufferEnd, FFeedbackContext* Warn) override;
	virtual bool FactoryCanImport(const FString& Filename) override;
	//~ End UFactory Interface
};
