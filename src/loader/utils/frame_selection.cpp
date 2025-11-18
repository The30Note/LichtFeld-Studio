#include "loader/utils/frame_selection.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <format>
#include <regex>

namespace gs::loader::utils {

    float compute_spatial_distance(const gs::Camera& cam1, const gs::Camera& cam2) {
        // Get camera positions (3D world coordinates)
        const auto& pos1 = cam1.cam_position();
        const auto& pos2 = cam2.cam_position();
        
        // Ensure positions are on CPU and accessible
        auto pos1_cpu = pos1.cpu();
        auto pos2_cpu = pos2.cpu();
        
        // Compute Euclidean distance: sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
        float dx = pos1_cpu[0].item<float>() - pos2_cpu[0].item<float>();
        float dy = pos1_cpu[1].item<float>() - pos2_cpu[1].item<float>();
        float dz = pos1_cpu[2].item<float>() - pos2_cpu[2].item<float>();
        
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    std::vector<std::shared_ptr<gs::Camera>> select_frames_spatial(
        const std::vector<std::shared_ptr<gs::Camera>>& cameras,
        float min_distance) {
        
        if (cameras.empty()) {
            return {};
        }
        
        std::vector<std::shared_ptr<gs::Camera>> selected;
        selected.reserve(cameras.size());
        
        // Always include the first camera
        selected.push_back(cameras[0]);
        
        // Add camera only if it's at least min_distance away from the last selected camera
        for (size_t i = 1; i < cameras.size(); ++i) {
            const auto& current_cam = cameras[i];
            const auto& last_selected = selected.back();
            
            float distance = compute_spatial_distance(*current_cam, *last_selected);
            
            if (distance >= min_distance) {
                selected.push_back(current_cam);
            }
        }
        
        return selected;
    }

    std::vector<std::shared_ptr<gs::Camera>> select_frames_temporal(
        const std::vector<std::shared_ptr<gs::Camera>>& cameras,
        int gap) {
        
        if (cameras.empty()) {
            return {};
        }
        
        std::vector<std::shared_ptr<gs::Camera>> selected;
        selected.reserve(cameras.size());
        
        // Always include the first camera
        selected.push_back(cameras[0]);
        int last_frame_id = cameras[0]->uid();
        
        // gap = 5 means keep every 5th frame (skip 4 frames between selected frames)
        // Use camera UID as the frame identifier
        for (size_t i = 1; i < cameras.size(); ++i) {
            const auto& current_cam = cameras[i];
            int current_frame_id = current_cam->uid();
            
            if (current_frame_id - last_frame_id >= gap) {
                selected.push_back(current_cam);
                last_frame_id = current_frame_id;
            }
        }
        
        return selected;
    }

    std::vector<std::shared_ptr<gs::Camera>> select_frames_combined(
        const std::vector<std::shared_ptr<gs::Camera>>& cameras,
        float min_spatial,
        int temporal_gap) {
        
        if (cameras.empty()) {
            return {};
        }
        
        std::vector<std::shared_ptr<gs::Camera>> selected;
        selected.reserve(cameras.size());
        
        // Always include the first camera
        selected.push_back(cameras[0]);
        int last_frame_id = cameras[0]->uid();
        
        // Both spatial and temporal criteria must be met
        // temporal_gap = 5 means keep every 5th frame (skip 4 frames between selected frames)
        // Use camera UID as the frame identifier
        for (size_t i = 1; i < cameras.size(); ++i) {
            const auto& current_cam = cameras[i];
            const auto& last_selected = selected.back();
            
            // Check spatial distance
            float spatial_dist = compute_spatial_distance(*current_cam, *last_selected);
            if (spatial_dist < min_spatial) {
                continue;
            }
            
            // Temporal gap using camera UID
            int current_frame_id = current_cam->uid();
            
            if (current_frame_id - last_frame_id >= temporal_gap) {
                selected.push_back(current_cam);
                last_frame_id = current_frame_id;
            }
        }
        
        return selected;
    }

    std::vector<std::shared_ptr<gs::Camera>> filter_cameras(
        const std::vector<std::shared_ptr<gs::Camera>>& cameras,
        const LoadOptions& options) {
        
        if (!options.enable_frame_selection || cameras.empty()) {
            return cameras; // Return original list if selection is disabled or empty
        }
        
        size_t original_size = cameras.size();
        std::vector<std::shared_ptr<gs::Camera>> filtered;
        
        switch (options.frame_selection_mode) {
            case FrameSelectionMode::Spatial:
                filtered = select_frames_spatial(cameras, options.min_spatial_distance);
                break;
                
            case FrameSelectionMode::Temporal:
                filtered = select_frames_temporal(cameras, options.temporal_gap);
                break;
                
            case FrameSelectionMode::Both:
                filtered = select_frames_combined(cameras, options.min_spatial_distance, options.temporal_gap);
                break;
                
            case FrameSelectionMode::None:
            default:
                return cameras;
        }
        
        size_t filtered_size = filtered.size();
        float percentage = (original_size > 0) ? (100.0f * filtered_size / original_size) : 0.0f;
        
        LOG_INFO("Frame selection: Selected {} cameras from {} ({:.1f}%)", 
                 filtered_size, original_size, percentage);
        
        return filtered;
    }

} // namespace gs::loader::utils

