<!--
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
-->

# Unreal® NNE Runtime RDG for ML Extensions for Vulkan®

## Introduction

This Unreal® Engine plugin extends Neural Network Engine (NNE) by providing a new runtime that enables support for ML Extensions for Vulkan®.

Specifically, the plugin implements the `INNERuntimeRDG` interface provided by the engine. This allows machine learning models authored by the ML SDK for Vulkan® to be run using the ML Extensions for Vulkan® as part of the Render Depdendency Graph (RDG), allowing for the tight integration of ML inference and rendering.

## System requirements

- Windows 11
- Unreal® Engine 5.5
- Visual Studio 2022, with the "Desktop Development with C++" and ".NET desktop build tools" packs enabled
- Vulkan® SDK

## Setup

1. **If you downloaded this plugin from the *GitHub release package*, then skip this step.**

   Run the `BuildThirdParty.ps1` script to download and build the required third-party dependencies. (These are included as part of the *GitHub release package*, but are not part of the Git repository as they are external dependencies.). You may need to install some build tools like `cmake` and `git`. 

2. Copy this folder into the `Plugins/` folder in either your Unreal® `Engine` folder or your project folder.

3. Enable the ML Emulation Layer for Vulkan® by following the instructions here: https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/nss-unreal/2-emulation-layer/. You can find the built emulation layers in the `MLEmulationLayerForVulkan` folder.

4. Enable the plugin (e.g. in your `.uproject` file or using the Plugins window).

5. Build your project and the engine if needed.

6. You should now have accesss to the `NNERuntimeRDGMLExtensionsForVulkan` NNE runtime which can be used with the NNE framework. One example of this being used is the *NSS* plugin.

## Trademarks and Copyrights

Arm® is a registered trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere.

Unreal® is a trademark or registered trademark of Epic Games, Inc. in the United States of America and elsewhere.

Vulkan is a registered trademark and the Vulkan SC logo is a trademark of the Khronos Group Inc.

Visual Studio, Windows are registered trademarks or trademarks of Microsoft Corporation in the US and other jurisdictions.
