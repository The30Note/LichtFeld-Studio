/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "loader/loader.hpp"
#include <memory>
#include <vector>

namespace gs::loader::utils {

    /**
     * @brief Compute 3D Euclidean distance between two camera positions
     * @param cam1 First camera
     * @param cam2 Second camera
     * @return Distance in world units
     */
    float compute_spatial_distance(const gs::Camera& cam1, const gs::Camera& cam2);

    /**
     * @brief Select frames using greedy spatial distance algorithm
     * @param cameras Input camera list
     * @param min_distance Minimum spatial distance between selected cameras
     * @return Filtered camera list
     */
    std::vector<std::shared_ptr<gs::Camera>> select_frames_spatial(
        const std::vector<std::shared_ptr<gs::Camera>>& cameras,
        float min_distance);

    /**
     * @brief Select frames using greedy temporal gap algorithm
     * @param cameras Input camera list
     * @param gap Gap between selected frames (e.g., 5 = keep every 5th frame, skipping 4 frames between)
     * @return Filtered camera list
     */
    std::vector<std::shared_ptr<gs::Camera>> select_frames_temporal(
        const std::vector<std::shared_ptr<gs::Camera>>& cameras,
        int gap);

    /**
     * @brief Select frames using combined spatial and temporal criteria
     * Both criteria must be met for a frame to be selected
     * @param cameras Input camera list
     * @param min_spatial Minimum spatial distance
     * @param temporal_gap Gap between selected frames (e.g., 5 = keep every 5th frame)
     * @return Filtered camera list
     */
    std::vector<std::shared_ptr<gs::Camera>> select_frames_combined(
        const std::vector<std::shared_ptr<gs::Camera>>& cameras,
        float min_spatial,
        int temporal_gap);

    /**
     * @brief Main entry point for frame selection based on LoadOptions
     * @param cameras Input camera list
     * @param options Load options containing frame selection parameters
     * @return Filtered camera list
     */
    std::vector<std::shared_ptr<gs::Camera>> filter_cameras(
        const std::vector<std::shared_ptr<gs::Camera>>& cameras,
        const LoadOptions& options);

} // namespace gs::loader::utils

