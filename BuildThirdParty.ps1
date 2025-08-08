# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

# This script builds the third-party dependencies for the plugin and copies them into the plugin

$ErrorActionPreference = "Stop"
$OriginalLocation = Get-Location # Remember the current directory so that we can restore it later in case of an error

# Helper function that checks the given exit code and throws an error if it is not zero.
function ThrowIfFailed {
    param(
        [int]$ExitCode,
        [string]$Step
    )

    if ($ExitCode -ne 0) {
        Write-Error "ERROR: $Step failed with exit code $ExitCode."
        Set-Location $OriginalLocation
        exit $ExitCode
    }
}

# Helper function that does a shallow clone of a repository at the specified commit (hash, branch or tag).
function CloneRepoFromCommit{
    param(
        [string]$RepoUrl,
        [string]$Commit,
        [string]$TargetDirName
    )

    git init $TargetDirName
    ThrowIfFailed $LASTEXITCODE "git init $TargetDirName"

    Push-Location $TargetDirName
    git remote add origin $RepoUrl
    ThrowIfFailed $LASTEXITCODE "git remote add origin $RepoUrl"

    git fetch --depth=1 origin $Commit
    ThrowIfFailed $LASTEXITCODE "git fetch --depth=1 origin $Commit"

    git checkout FETCH_HEAD
    ThrowIfFailed $LASTEXITCODE "git checkout FETCH_HEAD"

    Pop-Location
}

# Helper function that creates a temporary drive letter and maps it to the specified target path.
function New-TemporarySubstDrive {
    param (
        [string]$TargetPath
    )

    # Get list of used drive letters
    $used = Get-PSDrive -PSProvider FileSystem | ForEach-Object { $_.Name.ToUpper() }

    # Define candidate drive letters (Z to D)
    $candidates = [char[]]([char]'Z'..[char]'D')

    foreach ($letter in $candidates) {
        if ($used -notcontains $letter) {
            $drive = "$letter`:"
            subst $drive $TargetPath
            if ($LASTEXITCODE -eq 0) {
                Write-Output $drive
                return
            }
        }
    }

    throw "No available drive letter found for subst."
}


# Create and use a temporary folder for downloading and building
$WorkingDir = "$PSScriptRoot/Intermediate/ThirdPartyBuild"
if (Test-Path "$WorkingDir") {
    Remove-Item -Recurse -Force "$WorkingDir"
}
New-Item -ItemType Directory -Force -Path "$WorkingDir" | Out-Null
Push-Location "$WorkingDir"

# --- Clone repositories ---
Write-Host "Cloning ai-ml-sdk-vgf-library..."
CloneRepoFromCommit -RepoUrl "https://github.com/arm/ai-ml-sdk-vgf-library.git" -Commit "bf2de11731b7e60a8f1fe04da47a56fa003a80ed" -TargetDirName "ai-ml-sdk-vgf-library"

Write-Host "Cloning flatbuffers..."
CloneRepoFromCommit -RepoUrl "https://github.com/google/flatbuffers.git" -Commit "v23.5.26" -TargetDirName "flatbuffers"

Write-Host "Cloning SPIRV-Tools..."
CloneRepoFromCommit -RepoUrl "https://github.com/arm/SPIRV-Tools.git" -Commit "4ed8384fb5c356f2241a43b31f2d34fb82c1883d" -TargetDirName "SPIRV-Tools"

Write-Host "Cloning SPIRV-Headers..."
CloneRepoFromCommit -RepoUrl "https://github.com/arm/SPIRV-Headers.git" -Commit "de1807b7cfa8e722979d5ab7b7445b258dbc1836" -TargetDirName "SPIRV-Headers"

Write-Host "Cloning Vulkan-Headers..."
CloneRepoFromCommit -RepoUrl "https://github.com/KhronosGroup/Vulkan-Headers.git" -Commit "vulkan-sdk-1.4.321.0" -TargetDirName "Vulkan-Headers"

Write-Host "Cloning ai-ml-emulation-layer-for-vulkan..."
CloneRepoFromCommit -RepoUrl "https://github.com/arm/ai-ml-emulation-layer-for-vulkan.git" -Commit "788ac998abdf33df6efe71c5b0df365d2e721c3c" -TargetDirName "ai-ml-emulation-layer-for-vulkan"

Write-Host "Cloning glslang..."
CloneRepoFromCommit -RepoUrl "https://github.com/KhronosGroup/glslang.git" -Commit "vulkan-sdk-1.4.321.0" -TargetDirName "glslang"

Write-Host "Cloning SPIRV-Cross..."
CloneRepoFromCommit -RepoUrl "https://github.com/KhronosGroup/SPIRV-Cross.git" -Commit "256192e6248c8f6234323c998993126e841e1830" -TargetDirName "SPIRV-Cross"



# --- Build dependencies  ---

# Build .lib files that are compatible with Unreal's preferred toolchain
$COMMON_CMAKE_ARGS = @('-G', 'Visual Studio 17 2022', '-T', 'v143,version=14.38')

# The cmake builds can run into problems with long pathnames, so our working dir to a temporary drive
$TempDrive = New-TemporarySubstDrive "$WorkingDir"
Push-Location $TempDrive
try {
    Write-Host "Building ai-ml-sdk-vgf-library..."
    Push-Location "ai-ml-sdk-vgf-library"
    New-Item -ItemType Directory -Force -Path "Build" | Out-Null
    cmake -S "." -B "Build" @COMMON_CMAKE_ARGS -DML_SDK_VGF_LIB_BUILD_TOOLS=OFF -DFLATBUFFERS_PATH="../flatbuffers"
    ThrowIfFailed $LASTEXITCODE "CMake configure ai-ml-sdk-vgf-library"
    cmake --build "Build" --config Release --target "vgf"
    ThrowIfFailed $LASTEXITCODE "Build ai-ml-sdk-vgf-library"
    Pop-Location

    Write-Host "Building SPIRV-Tools..."
    Push-Location "SPIRV-Tools"
    New-Item -ItemType Directory -Force -Path "Build" | Out-Null
    $SPIRVHeadersPath = ("$PWD/../SPIRV-Headers").Replace("\", "/") # Backslashes cause problem in the cmake build script
    cmake -S "." -B "Build" @COMMON_CMAKE_ARGS -DSPIRV-Headers_SOURCE_DIR="$SPIRVHeadersPath"
    ThrowIfFailed $LASTEXITCODE "CMake configure SPIRV-Tools"
    cmake --build "Build" --config Release --target "spirv-opt"
    ThrowIfFailed $LASTEXITCODE "Build SPIRV-Tools"
    Pop-Location

    Write-Host "Building ai-ml-emulation-layer-for-vulkan..."
    Push-Location "ai-ml-emulation-layer-for-vulkan"
    New-Item -ItemType Directory -Force -Path "Build" | Out-Null
    cmake -S "." -B "Build" @COMMON_CMAKE_ARGS -DVMEL_TESTS_ENABLE=0 -DVULKAN_HEADERS_PATH="$TempDrive/vulkan-headers" -DSPIRV_TOOLS_PATH="$TempDrive/SPIRV-Tools" -DSPIRV_HEADERS_PATH="$TempDrive/SPIRV-Headers" -DSPIRV_CROSS_PATH="$TempDrive/SPIRV-Cross" -DGLSLANG_PATH="$TempDrive/glslang"
    ThrowIfFailed $LASTEXITCODE "CMake configure ai-ml-emulation-layer-for-vulkan"
    cmake --build "Build" --config Release --target "VkLayer_Tensor" "VkLayer_Graph"
    ThrowIfFailed $LASTEXITCODE "Build ai-ml-emulation-layer-for-vulkan"
    Pop-Location
}
finally {
    Pop-Location
    Write-Host "Removing subst drive $TempDrive"
    subst.exe $TempDrive /d | Out-Null
}

# --- Copy files to the plugin's ThirdParty folder ---
$DestThirdParty = "$PSScriptRoot/Source/ThirdParty"

Write-Host "Copying ai-ml-sdk-vgf-library includes..."
if (Test-Path "$DestThirdParty/AIMLSDKVGFLibrary/Include") {
    Remove-Item -Recurse -Force "$DestThirdParty/AIMLSDKVGFLibrary/Include"
}
Copy-Item -Recurse "ai-ml-sdk-vgf-library/include-c/" "$DestThirdParty/AIMLSDKVGFLibrary/Include"

Write-Host "Copying ai-ml-sdk-vgf-library libs..."`
if (Test-Path "$DestThirdParty/AIMLSDKVGFLibrary/Lib") {
    Remove-Item -Recurse -Force "$DestThirdParty/AIMLSDKVGFLibrary/Lib"
}
New-Item -ItemType Directory -Force -Path "$DestThirdParty/AIMLSDKVGFLibrary/Lib/Win64" | Out-Null
Copy-Item -Force "ai-ml-sdk-vgf-library/Build/src/Release/vgf.lib" "$DestThirdParty/AIMLSDKVGFLibrary/Lib/Win64/vgf.lib"

Write-Host "Copying SPIRV-Tools includes..."
if (Test-Path "$DestThirdParty/SPIRVTools/Include") {
    Remove-Item -Recurse -Force "$DestThirdParty/SPIRVTools/Include"
}
New-Item -ItemType Directory -Force -Path "$DestThirdParty/SPIRVTools/Include/spirv-tools" | Out-Null
Copy-Item -Recurse -Force "SPIRV-Tools/include/spirv-tools/libspirv.h" "$DestThirdParty/SPIRVTools/Include/spirv-tools/libspirv.h"
Copy-Item -Recurse -Force "SPIRV-Headers/include/spirv/unified1/spirv.hpp11" "$DestThirdParty/SPIRVTools/Include/spirv-tools/spirv.hpp11"

Write-Host "Copying SPIRV-Tools libs..."
if (Test-Path "$DestThirdParty/SPIRVTools/Lib") {
    Remove-Item -Recurse -Force "$DestThirdParty/SPIRVTools/Lib"
}
New-Item -ItemType Directory -Force -Path "$DestThirdParty/SPIRVTools/Lib/Win64" | Out-Null
Copy-Item -Recurse -Force "SPIRV-Tools/Build/source/opt/Release/SPIRV-Tools-opt.lib" "$DestThirdParty/SPIRVTools/Lib/Win64/SPIRV-Tools-opt.lib"
Copy-Item -Recurse -Force "SPIRV-Tools/Build/source/Release/SPIRV-Tools.lib" "$DestThirdParty/SPIRVTools/Lib/Win64/SPIRV-Tools.lib"

Write-Host "Copying Vulkan-Headers includes..."
if (Test-Path "$DestThirdParty/Vulkan/Include") {
    Remove-Item -Recurse -Force "$DestThirdParty/Vulkan/Include"
}
Copy-Item -Recurse -Force "Vulkan-Headers/include/" "$DestThirdParty/Vulkan/Include"

Write-Host "Copying ai-ml-emulation-layer-for-vulkan binaries..."
if (Test-Path "$PSScriptRoot/MLEmulationLayerForVulkan/") {
    Remove-Item -Recurse -Force "$PSScriptRoot/MLEmulationLayerForVulkan/"
}
New-Item -ItemType Directory -Force -Path "$PSScriptRoot/MLEmulationLayerForVulkan/" | Out-Null
Copy-Item -Recurse -Force "ai-ml-emulation-layer-for-vulkan/Build/tensor/Release/VkLayer_Tensor.dll" "$PSScriptRoot/MLEmulationLayerForVulkan/"
Copy-Item -Recurse -Force "ai-ml-emulation-layer-for-vulkan/Build/tensor/Release/VkLayer_Tensor.json" "$PSScriptRoot/MLEmulationLayerForVulkan/"
Copy-Item -Recurse -Force "ai-ml-emulation-layer-for-vulkan/Build/graph/Release/VkLayer_Graph.dll" "$PSScriptRoot/MLEmulationLayerForVulkan/"
Copy-Item -Recurse -Force "ai-ml-emulation-layer-for-vulkan/Build/graph/Release/VkLayer_Graph.json" "$PSScriptRoot/MLEmulationLayerForVulkan/"

Write-Host "Successfully built third-party dependencies for the plugin!" -ForegroundColor Green

Pop-Location




