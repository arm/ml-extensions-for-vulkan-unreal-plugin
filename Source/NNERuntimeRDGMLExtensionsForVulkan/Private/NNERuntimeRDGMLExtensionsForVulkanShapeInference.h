// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

// This file provides an interface to the third party SPIRV-Tools code, specifically the graph shape inference pass.

#pragma once

#include "Containers/Map.h"
#include "Containers/Array.h"
#include "Containers/ArrayView.h"

// Map from (descriptor set number and binding idx) to a tensor shape.
using FDescriptorSetBindingToShapeMap = TMap<TPair<uint32_t, uint32_t>, TArray<int64_t>>;

struct ShapeInferenceResults
{
	bool Success;
	FDescriptorSetBindingToShapeMap OutputShapes;
	TArray<uint32_t> NewCode;
};

// Performs shape inference on the graph contained in the given SPIR-V code, using the given input shapes and propagating 
// shape information through the graph. The result is new SPIR-V code that is fully shaped and a map of output tensor shapes
// indexed by their binding information.
ShapeInferenceResults RunShapeInference(TConstArrayView<uint32_t> Code, FDescriptorSetBindingToShapeMap InputShapes);