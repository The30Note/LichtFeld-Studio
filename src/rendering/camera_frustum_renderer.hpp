/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <filesystem>

namespace gs::rendering {

    class CameraFrustumRenderer {
    public:
        CameraFrustumRenderer() = default;
        ~CameraFrustumRenderer() = default;

        Result<void> init();
        Result<void> render(const std::vector<std::shared_ptr<const Camera>>& cameras,
                            const glm::mat4& view,
                            const glm::mat4& projection,
                            float scale = 0.1f,
                            const glm::vec4& wire_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            const glm::vec4& solid_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            const glm::mat4& world_transform = glm::mat4(1.0f),
                            bool show_images = false,
                            float image_opacity = 0.25f);

        Result<int> pickCamera(const std::vector<std::shared_ptr<const Camera>>& cameras,
                               const glm::vec2& mouse_pos,
                               const glm::vec2& viewport_pos,
                               const glm::vec2& viewport_size,
                               const glm::mat4& view,
                               const glm::mat4& projection,
                               float scale = 0.1f,
                               const glm::mat4& world_transform = glm::mat4(1.0f));

        void setHighlightedCamera(int index) { highlighted_camera_ = index; }
        int getHighlightedCamera() const { return highlighted_camera_; }

        bool isInitialized() const { return initialized_; }

    private:
        Result<void> createGeometry();
        Result<void> createPickingFBO();
        void prepareInstances(const std::vector<std::shared_ptr<const Camera>>& cameras,
                              float scale,
                              const glm::vec4& wire_color,
                              const glm::vec4& solid_color,
                              bool for_picking = false,
                              const glm::vec3& view_position = glm::vec3(0, 0, 0),
                              const glm::mat4& world_transform = glm::mat4(1.0f),
                              bool show_images = false);

        ManagedShader shader_;
        VAO vao_;
        VBO vbo_;
        VBO uv_vbo_;
        EBO face_ebo_;
        EBO edge_ebo_;
        VBO instance_vbo_;

        // Picking support
        FBO picking_fbo_;
        Texture picking_color_texture_;
        Texture picking_depth_texture_;
        int picking_fbo_width_ = 0;
        int picking_fbo_height_ = 0;

        // Camera tracking
        std::vector<int> camera_ids_;
        std::vector<glm::vec3> camera_positions_;
        int highlighted_camera_ = -1;

        size_t num_face_indices_ = 0;
        size_t num_edge_indices_ = 0;
        bool initialized_ = false;

        struct InstanceData {
            glm::mat4 transform;
            glm::vec4 color; // Changed to vec4 for RGBA (r, g, b, a)
            unsigned int texture_id; // Texture ID for camera image (0 = no texture)
            float aspect_ratio; // Image aspect ratio (width/height)
        };

        // Texture cache for camera images
        class CameraTextureCache {
        public:
            CameraTextureCache();
            ~CameraTextureCache();
            
            unsigned int getTexture(const std::filesystem::path& image_path);
            std::pair<int, int> getImageDimensions(const std::filesystem::path& image_path) const;
            void clear();

        private:
            struct CacheKey {
                std::filesystem::path image_path;
                
                bool operator==(const CacheKey& other) const {
                    return image_path == other.image_path;
                }
            };
            
            struct CacheKeyHash {
                std::size_t operator()(const CacheKey& key) const {
                    return std::hash<std::string>()(key.image_path.string());
                }
            };
            
            struct CacheEntry {
                unsigned int texture_id;
                int image_width;
                int image_height;
                std::chrono::steady_clock::time_point last_access;
            };
            
            std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> texture_cache_;
            static constexpr size_t MAX_CACHE_SIZE = 20;
            
            void evictOldest();
            unsigned int loadTexture(const std::filesystem::path& path);
        };

        CameraTextureCache texture_cache_;

        // Cached instances to avoid regeneration during picking
        std::vector<InstanceData> cached_instances_;

        // Track when cache needs update
        float last_scale_ = -1.0f;
        glm::vec4 last_wire_color_{-1, -1, -1, -1};
        glm::vec4 last_solid_color_{-1, -1, -1, -1};
        glm::vec3 last_view_position_{0, 0, 0};
        bool last_show_images_ = false;
        glm::mat4 last_world_transform_{1.0f}; // Track world transform to detect changes
    };

} // namespace gs::rendering