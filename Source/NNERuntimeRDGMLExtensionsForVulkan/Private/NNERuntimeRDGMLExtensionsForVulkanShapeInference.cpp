// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "NNERuntimeRDGMLExtensionsForVulkanShapeInference.h"
#include "NNERuntimeRDGMLExtensionsForVulkanModule.h"
#include "Algo/Transform.h"

#include "spirv-tools/libspirv.h"
#include "spirv-tools/spirv.hpp11"

namespace {

	template<typename T, void(*DeleterFuncPtr)(T*)>
	struct CustomDeleter
	{
		void operator()(T* Ptr) const
		{
			if (Ptr != nullptr)
			{
				DeleterFuncPtr(Ptr);
			}
		}
	};

	template<typename T, void(*DeleterFuncPtr)(T*)>
	using TUniquePtrWithCustomDeleter = TUniquePtr<T, CustomDeleter<T, DeleterFuncPtr>>;

	uint32_t GetSingleWordOperand(const spv_parsed_instruction_t& Instruction, int OperandIdx) {
		check(Instruction.operands[OperandIdx].num_words == 1);
		return Instruction.words[Instruction.operands[OperandIdx].offset];
	}

} // namespace

ShapeInferenceResults RunShapeInference(TConstArrayView<uint32_t> Code, FDescriptorSetBindingToShapeMap InputShapes)
{
	TUniquePtrWithCustomDeleter<spv_optimizer_t, spvOptimizerDestroy> Optimizer(spvOptimizerCreate(SPV_ENV_VULKAN_1_3));
	if (Optimizer == nullptr)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Failed to create spv_optimizer_t"));
		return ShapeInferenceResults{ false };
	}

	spvOptimizerSetMessageConsumer(Optimizer.Get(), [](spv_message_level_t Level, const char* Source, const spv_position_t* Position, const char* Message) {
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Display, TEXT("spvOptimizer: %hs"), Message);
	});

	TUniquePtrWithCustomDeleter<spv_optimizer_options_t, spvOptimizerOptionsDestroy> OptimizerOptions(spvOptimizerOptionsCreate());
	if (OptimizerOptions == nullptr)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Failed to create spv_optimizer_options_t"));
		return ShapeInferenceResults{ false };
	}

	TArray<spv_graph_shape_input> InputShapesForSpirv;
	Algo::Transform(InputShapes, InputShapesForSpirv, [](auto&& x) {
		spv_graph_shape_input input;
		input.descriptor_set = x.Get<0>().Get<0>();
		input.binding_id = x.Get<0>().Get<1>();
		input.rank = x.Get<1>().Num();
		input.shape = x.Get<1>().GetData();
		return input;
	});

	spvOptimizerRegisterGraphShapePass(Optimizer.Get(), InputShapesForSpirv.Num(), InputShapesForSpirv.GetData());

	spv_binary_t* OptimizedBinaryRaw = nullptr;
	spv_result_t Result = spvOptimizerRun(Optimizer.Get(), Code.GetData(), Code.Num(), &OptimizedBinaryRaw, OptimizerOptions.Get());
	TUniquePtrWithCustomDeleter<spv_binary_t, spvBinaryDestroy> OptimizedCode(OptimizedBinaryRaw);
	if (Result != SPV_SUCCESS)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Failed to run spvOptimizerRun, error: %i"), Result);
		return ShapeInferenceResults{ false };
	}

	// Parse the binary with the newly shaped graph, to extract the output tensor shapes
	TUniquePtrWithCustomDeleter<spv_context_t, spvContextDestroy> Context(spvContextCreate(SPV_ENV_VULKAN_1_3));
	if (Context == nullptr)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Failed to create spv_context_t"));
		return ShapeInferenceResults{ false };
	}

	// Struct that is passed into the parsing callback to record things we need later
	struct ParsingResultsStruct
	{
		TArray<spv_parsed_instruction_t> ParsedInstructions;
		TArray<TArray<spv_parsed_operand_t>> ParsedOperands; // First index is the instruction idx, second index is the operand idx within the instruction
		TMap<uint32_t, uint32_t> IdToDeclarationInstructionIdx;
		TMap<TPair<uint32_t, spv::Decoration>, uint32_t> IdToDecorations; // Map from (SPIR-V ID and decoration kind) to the decoration value
	} ParsingResults;

	auto ParsingCallback = [](void* UserData, const spv_parsed_instruction_t* Instruction) -> spv_result_t {
		ParsingResultsStruct& Results = *static_cast<ParsingResultsStruct*>(UserData);

		// Store the parsed instruction and its operands so that we can look back at them later.
		// Note that we need to copy the structs as they are transient
		Results.ParsedInstructions.Add(*Instruction);
		Results.ParsedOperands.Add(TArray<spv_parsed_operand_t>(Instruction->operands, Instruction->num_operands));
		// Update the copied instruction struct's pointer to point to our new copy, not the transient ones
		Results.ParsedInstructions.Last().operands = Results.ParsedOperands.Last().GetData();

		if (Instruction->result_id != 0)
		{
			Results.IdToDeclarationInstructionIdx.Add(Instruction->result_id, uint32_t(Results.ParsedInstructions.Num()) - 1);
		}
		if (Instruction->opcode == uint16_t(spv::Op::OpDecorate) &&
			Instruction->num_operands >= 3 && Instruction->operands[1].type == SPV_OPERAND_TYPE_DECORATION)
		{
			uint32 DecoratedId = GetSingleWordOperand(*Instruction, 0);
			spv::Decoration DecorationKind = spv::Decoration(GetSingleWordOperand(*Instruction, 1));
			uint32 DecorationValue = GetSingleWordOperand(*Instruction, 2);
			if (DecorationKind == spv::Decoration::Binding || DecorationKind == spv::Decoration::DescriptorSet)
			{
				Results.IdToDecorations.Add({ DecoratedId, DecorationKind }, DecorationValue);
			}
		}

		return SPV_SUCCESS;
	};

	spv_diagnostic_t* DiagnosticRaw = nullptr;
	Result = spvBinaryParse(Context.Get(), &ParsingResults, OptimizedCode->code, OptimizedCode->wordCount, nullptr, ParsingCallback, &DiagnosticRaw);
	TUniquePtrWithCustomDeleter<spv_diagnostic_t, spvDiagnosticDestroy> Diagnostic(DiagnosticRaw);
	if (Result != SPV_SUCCESS)
	{
		UE_LOG(LogNNERuntimeRDGMLExtensionsForVulkan, Error, TEXT("Failed to run spvBinaryParse, error code = %i, diagnostic = %hs"), Result, Diagnostic->error);
		return ShapeInferenceResults{ false };
	}

	ShapeInferenceResults Results;

	// Process the parsed instructions to extract the output shapes.
	for (int I = 0; I < ParsingResults.ParsedInstructions.Num(); ++I)
	{
		const spv_parsed_instruction_t& Instruction = ParsingResults.ParsedInstructions[I];
		if (Instruction.opcode == uint16_t(spv::Op::OpVariable))
		{
			uint32 VariableId = Instruction.result_id;

			// Find where the variable is declared
			const spv_parsed_instruction_t& VariableDeclaration = ParsingResults.ParsedInstructions[*ParsingResults.IdToDeclarationInstructionIdx.Find(VariableId)];
			// Find the type of the variable, which will always be OpTypePointer
			const spv_parsed_instruction_t& PointerTypeDeclaration = ParsingResults.ParsedInstructions[*ParsingResults.IdToDeclarationInstructionIdx.Find(VariableDeclaration.type_id)];
			// Find the type that the pointer points to
			const spv_parsed_instruction_t& TensorTypeDeclaration = ParsingResults.ParsedInstructions[*ParsingResults.IdToDeclarationInstructionIdx.Find(GetSingleWordOperand(PointerTypeDeclaration, 2))];
			if (TensorTypeDeclaration.opcode != uint32_t(spv::Op::OpTypeTensorARM))
			{
				continue;
			}

			// Get the tensor shape from the tensor type declaration
			const spv_parsed_instruction_t& ShapeDeclaration = ParsingResults.ParsedInstructions[*ParsingResults.IdToDeclarationInstructionIdx.Find(GetSingleWordOperand(TensorTypeDeclaration, 3))];
			if (ShapeDeclaration.opcode != uint32_t(spv::Op::OpConstantComposite))
			{
				continue;
			}
			TArray<int64_t> Shape;
			for (int Dim = 0; Dim < ShapeDeclaration.num_operands - 2; ++Dim)
			{
				uint32_t DimId = GetSingleWordOperand(ShapeDeclaration, Dim + 2);
				// Find the instruction which declares this dimension's value
				const spv_parsed_instruction_t& DimDeclaration = ParsingResults.ParsedInstructions[*ParsingResults.IdToDeclarationInstructionIdx.Find(DimId)];

				if (DimDeclaration.opcode !=uint32_t(spv::Op::OpConstant))
				{
					continue;
				}

				uint32_t DimValue = GetSingleWordOperand(DimDeclaration, 2);
				Shape.Add(static_cast<int64_t>(DimValue));
			}

			uint32 DescriptorSet = *ParsingResults.IdToDecorations.Find({ VariableId, spv::Decoration::DescriptorSet });
			uint32 BindingIdx = *ParsingResults.IdToDecorations.Find({ VariableId, spv::Decoration::Binding });
			Results.OutputShapes.Add({ DescriptorSet, BindingIdx }, MoveTemp(Shape));
		}
	}

	Results.NewCode = TArray<uint32_t>(OptimizedCode->code, OptimizedCode->wordCount);
	Results.Success = true;
	return Results;
}
