/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_frustum_renderer.hpp"
#include "core/logger.hpp"
#include "core/image_io.hpp"
#include "gl_state_guard.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glad/glad.h>
#include <cmath>

namespace gs::rendering {

    static constexpr float WIRE_THICKNESS = 1.0f;

    // CameraTextureCache Implementation
    CameraFrustumRenderer::CameraTextureCache::CameraTextureCache() {
        LOG_DEBUG("CameraTextureCache created");
    }

    CameraFrustumRenderer::CameraTextureCache::~CameraTextureCache() {
        clear();
    }

    void CameraFrustumRenderer::CameraTextureCache::clear() {
        for (auto& [key, entry] : texture_cache_) {
            if (entry.texture_id > 0) {
                glDeleteTextures(1, &entry.texture_id);
                entry.texture_id = 0; // Reset to prevent double-free
            }
        }
        texture_cache_.clear();
        LOG_DEBUG("CameraTextureCache cleared");
    }

    unsigned int CameraFrustumRenderer::CameraTextureCache::getTexture(const std::filesystem::path& image_path) {
        CacheKey key{image_path};

        if (auto it = texture_cache_.find(key); it != texture_cache_.end()) {
            it->second.last_access = std::chrono::steady_clock::now();
            LOG_TRACE("Camera texture cache hit for image {}", image_path.filename().string());
            return it->second.texture_id;
        }

        if (texture_cache_.size() >= MAX_CACHE_SIZE) {
            evictOldest();
        }

        auto [data_check, width, height, channels_check] = load_image(image_path);
        if (!data_check) {
            LOG_ERROR("Failed to load image data for dimensions: {}", image_path.string());
            return 0;
        }
        free_image(data_check);
        
        LOG_DEBUG("Loading camera image: {}", image_path.string());
        unsigned int texture_id = loadTexture(image_path);

        if (texture_id == 0) {
            LOG_ERROR("Failed to load camera texture from {}", image_path.string());
            return 0;
        }
        
        texture_cache_[key] = {texture_id, width, height, std::chrono::steady_clock::now()};
        LOG_DEBUG("Cached camera texture {} for image {} ({}x{})", texture_id, image_path.filename().string(), width, height);

        return texture_id;
    }
    
    std::pair<int, int> CameraFrustumRenderer::CameraTextureCache::getImageDimensions(const std::filesystem::path& image_path) const {
        CacheKey key{image_path};
        auto it = texture_cache_.find(key);
        if (it != texture_cache_.end()) {
            return {it->second.image_width, it->second.image_height};
        }
        return {0, 0};
    }

    void CameraFrustumRenderer::CameraTextureCache::evictOldest() {
        if (texture_cache_.empty())
            return;

        auto oldest = texture_cache_.begin();
        auto oldest_time = oldest->second.last_access;

        for (auto it = texture_cache_.begin(); it != texture_cache_.end(); ++it) {
            if (it->second.last_access < oldest_time) {
                oldest = it;
                oldest_time = it->second.last_access;
            }
        }

        LOG_TRACE("Evicting camera texture for image {} from cache", oldest->first.image_path.filename().string());
        if (oldest->second.texture_id > 0) {
            glDeleteTextures(1, &oldest->second.texture_id);
            oldest->second.texture_id = 0; // Reset to prevent double-free
        }
        texture_cache_.erase(oldest);
    }

    unsigned int CameraFrustumRenderer::CameraTextureCache::loadTexture(const std::filesystem::path& path) {
        if (!std::filesystem::exists(path)) {
            LOG_ERROR("Camera image file does not exist: {}", path.string());
            return 0;
        }

        try {
            auto [data, width, height, channels] = load_image(path);

            if (!data) {
                LOG_ERROR("Failed to load image data: {}", path.string());
                return 0;
            }

            LOG_TRACE("Loaded camera image: {}x{} with {} channels", width, height, channels);

            // FLIP vertically: OpenGL expects origin at bottom-left, images have origin at top-left
            std::vector<unsigned char> flipped_data(width * height * channels);
            size_t row_size = width * channels;
            for (int y = 0; y < height; ++y) {
                std::memcpy(
                    flipped_data.data() + y * row_size,
                    data + (height - 1 - y) * row_size,
                    row_size);
            }

            // Create OpenGL texture
            unsigned int texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);

            // Determine format based on channels
            GLenum format = GL_RGB;
            GLenum internal_format = GL_RGB8;

            if (channels == 1) {
                format = GL_RED;
                internal_format = GL_R8;
            } else if (channels == 2) {
                format = GL_RG;
                internal_format = GL_RG8;
            } else if (channels == 3) {
                format = GL_RGB;
                internal_format = GL_RGB8;
            } else if (channels == 4) {
                format = GL_RGBA;
                internal_format = GL_RGBA8;
            }

            // Upload flipped texture data
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0,
                         format, GL_UNSIGNED_BYTE, flipped_data.data());

            // Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            // Generate mipmaps for better quality when scaled
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

            // Free image data
            free_image(data);

            LOG_DEBUG("Created GL texture {} for camera image: {} ({}x{})",
                      texture, path.filename().string(), width, height);
            return texture;

        } catch (const std::exception& e) {
            LOG_ERROR("Exception loading camera image {}: {}", path.string(), e.what());
            return 0;
        }
    }

    Result<void> CameraFrustumRenderer::init() {
        LOG_DEBUG("Initializing camera frustum renderer");

        // Load shader
        auto shader_result = load_shader("camera_frustum", "camera_frustum.vert", "camera_frustum.frag", false);
        if (!shader_result) {
            LOG_ERROR("Failed to load camera frustum shader: {}", shader_result.error().what());
            return std::unexpected(shader_result.error().what());
        }
        shader_ = std::move(*shader_result);

        // Create geometry
        if (auto result = createGeometry(); !result) {
            return result;
        }

        // Create instance buffer
        auto instance_vbo_result = create_vbo();
        if (!instance_vbo_result) {
            return std::unexpected(instance_vbo_result.error());
        }
        instance_vbo_ = std::move(*instance_vbo_result);

        // Create picking FBO
        if (auto result = createPickingFBO(); !result) {
            return result;
        }

        initialized_ = true;
        LOG_INFO("Camera frustum renderer initialized");
        return {};
    }

    Result<void> CameraFrustumRenderer::createGeometry() {
        LOG_TIMER_TRACE("CameraFrustumRenderer::createGeometry");

        // Frustum vertices in camera space (apex at origin, base at z=-1)
        std::vector<glm::vec3> vertices = {
            // Base vertices (at z = -1, sized by FOV)
            {-0.5f, -0.5f, -1.0f}, // 0 bottom-left
            {0.5f, -0.5f, -1.0f},  // 1 bottom-right
            {0.5f, 0.5f, -1.0f},   // 2 top-right
            {-0.5f, 0.5f, -1.0f},  // 3 top-left
            // Apex (camera position)
            {0.0f, 0.0f, 0.0f} // 4
        };

        // UV coordinates for texture mapping (base plane only)
        std::vector<glm::vec2> uv_coords = {
            // Base vertices - map texture to base plane
            {0.0f, 0.0f}, // 0 bottom-left
            {1.0f, 0.0f}, // 1 bottom-right
            {1.0f, 1.0f}, // 2 top-right
            {0.0f, 1.0f}, // 3 top-left
            // Apex - no texture
            {0.5f, 0.5f}  // 4 (center, not used for texture)
        };

        // Face indices (triangles)
        std::vector<unsigned int> face_indices = {
            // Base (facing away)
            0, 1, 2,
            0, 2, 3,
            // Side faces
            0, 4, 1,
            1, 4, 2,
            2, 4, 3,
            3, 4, 0};

        // Edge indices (lines)
        std::vector<unsigned int> edge_indices = {
            0, 1, 1, 2, 2, 3, 3, 0, // Base edges
            0, 4, 1, 4, 2, 4, 3, 4  // Apex edges
        };

        num_face_indices_ = face_indices.size();
        num_edge_indices_ = edge_indices.size();

        // Create VAO and buffers
        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected(vao_result.error());
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected(vbo_result.error());
        }
        vbo_ = std::move(*vbo_result);

        auto uv_vbo_result = create_vbo();
        if (!uv_vbo_result) {
            return std::unexpected(uv_vbo_result.error());
        }
        uv_vbo_ = std::move(*uv_vbo_result);

        auto face_ebo_result = create_vbo();
        if (!face_ebo_result) {
            return std::unexpected(face_ebo_result.error());
        }
        face_ebo_ = std::move(*face_ebo_result);

        auto edge_ebo_result = create_vbo();
        if (!edge_ebo_result) {
            return std::unexpected(edge_ebo_result.error());
        }
        edge_ebo_ = std::move(*edge_ebo_result);

        // Build VAO
        VAOBuilder builder(std::move(*vao_result));

        // Vertex positions (location 0)
        std::span<const float> vertices_data(
            reinterpret_cast<const float*>(vertices.data()),
            vertices.size() * 3);

        builder.attachVBO(vbo_, vertices_data, GL_STATIC_DRAW)
            .setAttribute({.index = 0, .size = 3, .type = GL_FLOAT});

        // UV coordinates (location 1)
        std::span<const float> uv_data(
            reinterpret_cast<const float*>(uv_coords.data()),
            uv_coords.size() * 2);

        builder.attachVBO(uv_vbo_, uv_data, GL_STATIC_DRAW)
            .setAttribute({.index = 1, .size = 2, .type = GL_FLOAT});

        // Face indices
        builder.attachEBO(face_ebo_, std::span(face_indices), GL_STATIC_DRAW);

        vao_ = builder.build();

        // Also upload edge indices
        BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(edge_ebo_);
        upload_buffer(GL_ELEMENT_ARRAY_BUFFER, std::span(edge_indices), GL_STATIC_DRAW);

        LOG_DEBUG("Camera frustum geometry created");
        return {};
    }

    Result<void> CameraFrustumRenderer::createPickingFBO() {
        LOG_DEBUG("Creating picking framebuffer");

        // Create FBO
        GLuint fbo_id;
        glGenFramebuffers(1, &fbo_id);
        if (fbo_id == 0) {
            return std::unexpected("Failed to create picking FBO");
        }
        picking_fbo_ = FBO(fbo_id);

        // Initial size (will resize on first use)
        picking_fbo_width_ = 256;
        picking_fbo_height_ = 256;

        // Create color texture
        GLuint color_tex;
        glGenTextures(1, &color_tex);
        picking_color_texture_ = Texture(color_tex);

        glBindTexture(GL_TEXTURE_2D, picking_color_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, picking_fbo_width_, picking_fbo_height_,
                     0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Create depth texture
        GLuint depth_tex;
        glGenTextures(1, &depth_tex);
        picking_depth_texture_ = Texture(depth_tex);

        glBindTexture(GL_TEXTURE_2D, picking_depth_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, picking_fbo_width_, picking_fbo_height_,
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Attach to FBO
        glBindFramebuffer(GL_FRAMEBUFFER, picking_fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, picking_color_texture_, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, picking_depth_texture_, 0);

        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            LOG_ERROR("Picking FBO incomplete: 0x{:x}", status);
            return std::unexpected("Picking FBO incomplete");
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        LOG_DEBUG("Picking FBO created successfully");
        return {};
    }

    void CameraFrustumRenderer::prepareInstances(const std::vector<std::shared_ptr<const Camera>>& cameras,
                                                 float scale,
                                                 const glm::vec4& wire_color,
                                                 const glm::vec4& solid_color,
                                                 bool for_picking,
                                                 const glm::vec3& view_position,
                                                 const glm::mat4& world_transform,
                                                 bool show_images) {

        // Track if we need to regenerate
        bool needs_regeneration = false;

        // Check if we need to regenerate instances
        // Compare world transform matrices (with epsilon tolerance for floating point comparison)
        bool world_transform_changed = false;
        constexpr float transform_epsilon = 1e-6f;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (std::abs(world_transform[i][j] - last_world_transform_[i][j]) > transform_epsilon) {
                    world_transform_changed = true;
                    break;
                }
            }
            if (world_transform_changed) break;
        }
        
        if (cached_instances_.size() != cameras.size()) {
            needs_regeneration = true;
            LOG_TRACE("Instance count changed: {} -> {}", cached_instances_.size(), cameras.size());
        } else if (last_scale_ != scale || last_wire_color_ != wire_color || last_solid_color_ != solid_color ||
                   last_show_images_ != show_images || world_transform_changed) {
            needs_regeneration = true;
            if (world_transform_changed) {
                LOG_TRACE("World transform changed - regenerating instances to apply rotation and translation");
            } else {
                LOG_TRACE("Instance parameters changed");
            }
        } else if (!camera_ids_.empty() && camera_ids_.size() == cameras.size()) {
            // Check if camera IDs match (order matters for texture mapping)
            bool camera_order_matches = true;
            for (size_t i = 0; i < cameras.size(); ++i) {
                if (camera_ids_[i] != cameras[i]->uid()) {
                    camera_order_matches = false;
                    break;
                }
            }
            if (!camera_order_matches) {
                needs_regeneration = true;
                LOG_TRACE("Camera order changed - regenerating instances to fix texture mapping");
            }
        }

        // Only regenerate if necessary
        if (!needs_regeneration && !cached_instances_.empty()) {
            last_view_position_ = view_position;
            LOG_TRACE("Using {} cached instances for {}, updating visibility",
                      cached_instances_.size(), for_picking ? "picking" : "rendering");
            return;
        }

        LOG_DEBUG("Regenerating {} instances for {} (scale: {}, solid_color: [{}, {}, {}])",
                  cameras.size(), for_picking ? "picking" : "rendering", scale,
                  solid_color.r, solid_color.g, solid_color.b);

        cached_instances_.clear();
        cached_instances_.reserve(cameras.size());
        camera_ids_.clear();
        camera_ids_.reserve(cameras.size());
        camera_positions_.clear();
        camera_positions_.reserve(cameras.size());

        // Transform from OpenGL to COLMAP coordinates
        const glm::mat4 GL_TO_COLMAP = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));

        for (const auto& cam : cameras) {
            // Get camera world-to-camera transform
            auto R_tensor = cam->R();
            auto T_tensor = cam->T();

            if (!R_tensor.defined() || !T_tensor.defined()) {
                continue;
            }

            // Convert to CPU
            R_tensor = R_tensor.to(torch::kCPU);
            T_tensor = T_tensor.to(torch::kCPU);

            // Build world-to-camera matrix
            glm::mat4 w2c(1.0f);
            auto R_acc = R_tensor.accessor<float, 2>();
            auto T_acc = T_tensor.accessor<float, 1>();

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    w2c[j][i] = R_acc[i][j]; // Column-major
                }
                w2c[3][i] = T_acc[i];
            }

            glm::mat4 c2w = glm::inverse(w2c);
            glm::mat4 transformed_c2w = world_transform * c2w;

            glm::vec3 cam_pos = glm::vec3(transformed_c2w[3]);
            camera_positions_.push_back(cam_pos);

            glm::mat4 model = transformed_c2w * GL_TO_COLMAP * glm::scale(glm::mat4(1.0f), glm::vec3(scale));

            glm::vec4 color = solid_color;

            // Get camera FOV and calculate frustum base size
            // Frustum base is at z=-1 (distance 1.0 from camera)
            // Base size = 2 * distance * tan(FOV/2)
            float fov_x = cam->FoVx(); // Horizontal FOV in radians
            float fov_y = cam->FoVy(); // Vertical FOV in radians
            
            // Calculate base dimensions based on FOV
            // At distance 1.0, width = 2 * tan(fov_x / 2), height = 2 * tan(fov_y / 2)
            float base_width = 2.0f * std::tan(fov_x * 0.5f);
            float base_height = 2.0f * std::tan(fov_y * 0.5f);
            
            float aspect_ratio = base_width / base_height;

            unsigned int texture_id = 0;

            // Scale frustum base to match camera FOV
            // Base plane geometry is -0.5 to 0.5 (size 1.0), scale by FOV-based dimensions
            glm::mat4 fov_scale = glm::scale(glm::mat4(1.0f), glm::vec3(base_width, base_height, 1.0f));
            glm::mat4 scaled_model = model * fov_scale;

            cached_instances_.push_back({scaled_model, color, texture_id, aspect_ratio});
            camera_ids_.push_back(cam->uid());
        }

        // Update cache parameters
        last_scale_ = scale;
        last_wire_color_ = wire_color;
        last_solid_color_ = solid_color;
        last_view_position_ = view_position;
        last_show_images_ = show_images;
        last_world_transform_ = world_transform;

        LOG_DEBUG("Prepared {} instances", cached_instances_.size());
    }

    Result<void> CameraFrustumRenderer::render(
        const std::vector<std::shared_ptr<const Camera>>& cameras,
        const glm::mat4& view,
        const glm::mat4& projection,
        float scale,
        const glm::vec4& wire_color,
        const glm::vec4& solid_color,
        const glm::mat4& world_transform,
        bool show_images,
        float image_opacity) {

        if (!initialized_ || cameras.empty()) {
            return {};
        }

        LOG_TRACE("Rendering {} camera frustums (show_images: {}, opacity: {})", 
                  cameras.size(), show_images, image_opacity);

        glm::vec3 view_position = glm::vec3(glm::inverse(view)[3]);

        // Sort cameras by image path filename
        std::vector<std::shared_ptr<const Camera>> sorted_cameras = cameras;
        std::ranges::sort(sorted_cameras, [](const auto& a, const auto& b) {
            return a->image_path().filename() < b->image_path().filename();
        });

        // Prepare instance data for rendering (not picking) using sorted cameras
        prepareInstances(sorted_cameras, scale, wire_color, solid_color, false, view_position, world_transform, show_images);

        if (cached_instances_.empty()) {
            return {};
        }

        // Filter out instances with alpha = 0 for rendering
        std::vector<InstanceData> visible_instances;
        std::vector<int> visible_indices;
        visible_instances.reserve(cached_instances_.size());
        visible_indices.reserve(cached_instances_.size());

        for (size_t i = 0; i < cached_instances_.size(); ++i) {
            if (cached_instances_[i].color.a > 0.01f) { // Skip nearly invisible frustums
                visible_instances.push_back(cached_instances_[i]);
                visible_indices.push_back(static_cast<int>(i));
            }
        }

        // Use comprehensive state guard for entire render operation
        GLStateGuard state_guard;

        while (glGetError() != GL_NO_ERROR) {}

        // Bind shader using RAII
        {
            ShaderScope shader(shader_);

            if (!shader.isBound()) {
                LOG_ERROR("Failed to bind camera frustum shader");
                return std::unexpected("Failed to bind camera frustum shader");
            }

            // Set uniforms while shader is bound
            glm::mat4 view_proj = projection * view;

            // Set required uniforms
            if (auto result = shader->set("viewProj", view_proj); !result) {
                LOG_ERROR("Failed to set viewProj uniform: {}", result.error());
            }

            if (auto result = shader->set("viewPos", view_position); !result) {
                LOG_ERROR("Failed to set viewPos uniform: {}", result.error());
            }

            // Set picking mode to false for normal rendering
            if (auto result = shader->set("pickingMode", false); !result) {
                LOG_TRACE("pickingMode uniform not found");
            }


            int visible_highlight_index = -1;
            if (highlighted_camera_ >= 0 && highlighted_camera_ < static_cast<int>(cameras.size())) {
                // Get the camera ID from the original array
                int highlighted_camera_id = cameras[highlighted_camera_]->uid();
                
                // Find this camera in the sorted array
                int sorted_index = -1;
                for (size_t i = 0; i < sorted_cameras.size(); ++i) {
                    if (sorted_cameras[i]->uid() == highlighted_camera_id) {
                        sorted_index = static_cast<int>(i);
                        break;
                    }
                }
                
                // Now find this sorted index in the visible_indices array
                if (sorted_index >= 0) {
                    for (size_t i = 0; i < visible_indices.size(); ++i) {
                        if (visible_indices[i] == sorted_index) {
                            visible_highlight_index = static_cast<int>(i);
                            break;
                        }
                    }
                }
            }

            // Set highlight index
            if (auto result = shader->set("highlightIndex", visible_highlight_index); !result) {
                LOG_TRACE("highlightIndex uniform not found");
            }

            if (auto result = shader->set("highlightColor", glm::vec3(1.0f, 0.85f, 0.0f)); !result) {
                LOG_TRACE("highlightColor uniform not found");
            }

            // Set image-related uniforms
            if (auto result = shader->set("showImages", show_images); !result) {
                LOG_TRACE("showImages uniform not found");
            }
            if (auto result = shader->set("imageOpacity", image_opacity); !result) {
                LOG_TRACE("imageOpacity uniform not found");
            }
            // Set texture unit (GL_TEXTURE0 = 0)
            if (auto result = shader->set("cameraTexture", 0); !result) {
                LOG_TRACE("cameraTexture uniform not found");
            }
            
            // Set wire color for wireframe edges
            if (auto result = shader->set("wireColor", wire_color); !result) {
                LOG_TRACE("wireColor uniform not found");
            }
            
            // Default to solid rendering mode
            if (auto result = shader->set("wireframeMode", false); !result) {
                LOG_TRACE("wireframeMode uniform not found");
            }

            // Bind VAO using RAII
            {
                VAOBinder vao_bind(vao_);

                // Upload visible instance data using RAII buffer binding
                {
                    BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
                    upload_buffer(GL_ARRAY_BUFFER, std::span(visible_instances), GL_DYNAMIC_DRAW);

                    // Setup instance attributes while buffer is bound
                    // Instance transform matrix (locations 2-5, since 0=pos, 1=UV)
                    for (int i = 0; i < 4; ++i) {
                        glEnableVertexAttribArray(2 + i);
                        glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                              reinterpret_cast<void*>(sizeof(glm::vec4) * i));
                        glVertexAttribDivisor(2 + i, 1);
                    }

                    // Instance color RGBA (location 6) - vec4
                    glEnableVertexAttribArray(6);
                    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                          reinterpret_cast<void*>(offsetof(InstanceData, color)));
                    glVertexAttribDivisor(6, 1);

                    // Instance texture ID (location 7) - use IPointer for integer attribute
                    glEnableVertexAttribArray(7);
                    glVertexAttribIPointer(7, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                           reinterpret_cast<void*>(offsetof(InstanceData, texture_id)));
                    glVertexAttribDivisor(7, 1);
                    
                    // Instance aspect ratio (location 8)
                    glEnableVertexAttribArray(8);
                    glVertexAttribPointer(8, 1, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                          reinterpret_cast<void*>(offsetof(InstanceData, aspect_ratio)));
                    glVertexAttribDivisor(8, 1);
                }

                // Setup render state
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LESS);
                glDepthMask(GL_TRUE);
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

                // First pass: Render images if enabled
                if (show_images) {
                    if (auto result = shader->set("imageOpacity", image_opacity); !result) {
                        LOG_ERROR("Failed to set imageOpacity uniform: {}", result.error());
                    }
                    
                    std::vector<InstanceData> image_instances = visible_instances;
                    for (auto& inst : image_instances) {
                        inst.color.a = 1.0f;
                    }

                    // Load textures and update instance data with texture IDs
                    for (size_t i = 0; i < image_instances.size(); ++i) {
                        int original_index = visible_indices[i];
                        if (original_index < 0 || static_cast<size_t>(original_index) >= sorted_cameras.size()) {
                            continue;
                        }
                        
                        // Check distance from view position to camera position
                        constexpr float IMAGE_NEAR_DISTANCE = 0.005f;
                        if (static_cast<size_t>(original_index) < camera_positions_.size()) {
                            float distance = glm::distance(view_position, camera_positions_[original_index]);
                            if (distance > IMAGE_NEAR_DISTANCE) {
                                // Skip this image - too far away
                                image_instances[i].texture_id = 0;
                                continue;
                            }
                        }
                        
                        const auto& cam = sorted_cameras[original_index];
                        if (cam && show_images) {
                            unsigned int texture_id = texture_cache_.getTexture(cam->image_path());
                            image_instances[i].texture_id = texture_id;
                        }
                    }
                    
                    // Upload updated instances with texture IDs
                    {
                        BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
                        upload_buffer(GL_ARRAY_BUFFER, std::span(image_instances), GL_DYNAMIC_DRAW);
                    }

                    BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(face_ebo_);
                    
                    int textures_rendered = 0;
                    for (size_t i = 0; i < image_instances.size(); ++i) {
                        unsigned int texture_id = image_instances[i].texture_id;
                        
                        // Render base plane (first 6 indices) with texture if available
                        if (texture_id > 0) {
                            glActiveTexture(GL_TEXTURE0);
                            glBindTexture(GL_TEXTURE_2D, texture_id);
                            // Draw base plane only (first 2 triangles = 6 indices)
                            glDrawElementsInstancedBaseInstance(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, 1, static_cast<GLuint>(i));
                            textures_rendered++;
                        }
                    }
                    if (textures_rendered > 0) {
                        LOG_TRACE("Rendered {} camera frustums with textures", textures_rendered);
                    }
                    glBindTexture(GL_TEXTURE_2D, 0);

                    GLenum err = glGetError();
                    if (err != GL_NO_ERROR) {
                        LOG_ERROR("OpenGL error after drawing textured base planes: 0x{:x}", err);
                    }
                    
                    // Restore original instances for frustum rendering
                    {
                        BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
                        upload_buffer(GL_ARRAY_BUFFER, std::span(visible_instances), GL_DYNAMIC_DRAW);
                    }
                }

                if (auto result = shader->set("wireframeMode", false); !result) {
                    LOG_TRACE("wireframeMode uniform not found");
                }
                
                BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(face_ebo_);
                
                // Render side faces only (skip base plane - first 6 indices)
                for (size_t i = 0; i < visible_instances.size(); ++i) {
                    glDrawElementsInstancedBaseInstance(GL_TRIANGLES, num_face_indices_ - 6, GL_UNSIGNED_INT, 
                                                       reinterpret_cast<void*>(6 * sizeof(unsigned int)), 1, static_cast<GLuint>(i));
                }

                // Check for errors after drawing faces
                GLenum err = glGetError();
                if (err != GL_NO_ERROR) {
                    LOG_ERROR("OpenGL error after drawing faces: 0x{:x}", err);
                }

                // Second pass: wireframe edges on top
                // Use depth testing with a small bias so wires render through solid faces
                // but are still occluded by images (which were rendered first)
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LEQUAL);
                // Use polygon offset for lines to push them slightly forward in depth
                glEnable(GL_POLYGON_OFFSET_LINE);
                glPolygonOffset(0.0f, -1.0f);
                glLineWidth(WIRE_THICKNESS);
                
                // Enable wireframe mode to use wire color
                if (auto result = shader->set("wireframeMode", true); !result) {
                    LOG_TRACE("wireframeMode uniform not found");
                }

                {
                    BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(edge_ebo_);
                    glDrawElementsInstanced(GL_LINES, num_edge_indices_, GL_UNSIGNED_INT, 0, visible_instances.size());
                }
                
                // Restore solid rendering mode
                if (auto result = shader->set("wireframeMode", false); !result) {
                    LOG_TRACE("wireframeMode uniform not found");
                }
                
                // Restore depth state for subsequent rendering
                glDisable(GL_POLYGON_OFFSET_LINE);
                glDepthFunc(GL_LESS);

                // Check for errors after draw
                err = glGetError();
                if (err != GL_NO_ERROR) {
                    LOG_ERROR("OpenGL error after drawing: 0x{:x}", err);
                }

                // Cleanup instance attributes before VAO unbinds
                for (int i = 2; i <= 8; ++i) {
                    glDisableVertexAttribArray(i);
                    if (i >= 2 && i <= 8) {
                        glVertexAttribDivisor(i, 0);
                    }
                }
            } 
        }

        LOG_TRACE("Rendered {} camera frustums", visible_instances.size());
        return {};
    }

    Result<int> CameraFrustumRenderer::pickCamera(const std::vector<std::shared_ptr<const Camera>>& cameras,
                                                  const glm::vec2& mouse_pos,
                                                  const glm::vec2& viewport_pos,
                                                  const glm::vec2& viewport_size,
                                                  const glm::mat4& view,
                                                  const glm::mat4& projection,
                                                  float scale,
                                                  const glm::mat4& world_transform) {
        if (!initialized_ || cameras.empty()) {
            return -1;
        }

        // Use cached instances if available, don't regenerate!
        if (cached_instances_.empty() || camera_ids_.size() != cameras.size()) {
            // Only regenerate if we really have to (first pick or camera count changed)
            LOG_WARN("No cached instances for picking, regenerating");

            // Extract view position for visibility calculation
            glm::vec3 view_position = glm::vec3(glm::inverse(view)[3]);

            // Use the same colors as last render to avoid visual changes
            prepareInstances(cameras, scale, last_wire_color_, last_solid_color_, true, view_position, world_transform, false);

            if (cached_instances_.empty()) {
                LOG_ERROR("Failed to prepare instances for picking");
                return -1;
            }
        } else {
            LOG_TRACE("Using {} cached instances for picking", cached_instances_.size());
        }

        // Resize picking FBO if needed
        int vp_width = static_cast<int>(viewport_size.x);
        int vp_height = static_cast<int>(viewport_size.y);

        if (vp_width != picking_fbo_width_ || vp_height != picking_fbo_height_) {
            LOG_DEBUG("Resizing picking FBO from {}x{} to {}x{}",
                      picking_fbo_width_, picking_fbo_height_, vp_width, vp_height);
            picking_fbo_width_ = vp_width;
            picking_fbo_height_ = vp_height;

            glBindTexture(GL_TEXTURE_2D, picking_color_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, picking_fbo_width_, picking_fbo_height_,
                         0, GL_RGB, GL_FLOAT, nullptr);

            glBindTexture(GL_TEXTURE_2D, picking_depth_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, picking_fbo_width_, picking_fbo_height_,
                         0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        }

        // Save current FBO and viewport
        GLint current_fbo;
        GLint current_viewport[4];
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);
        glGetIntegerv(GL_VIEWPORT, current_viewport);

        // Bind picking FBO
        glBindFramebuffer(GL_FRAMEBUFFER, picking_fbo_);
        glViewport(0, 0, picking_fbo_width_, picking_fbo_height_);

        // Clear to black (ID = 0)
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render with picking shader
        {
            ShaderScope shader(shader_);

            if (!shader.isBound()) {
                glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
                glViewport(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);
                return std::unexpected("Failed to bind picking shader");
            }

            glm::mat4 view_proj = projection * view;
            glm::vec3 view_pos = glm::vec3(glm::inverse(view)[3]);

            shader->set("viewProj", view_proj);
            shader->set("viewPos", view_pos);
            shader->set("pickingMode", true);

            // Set minimum pick distance based on scale - don't pick frustums too close
            float min_pick_distance = scale * 2.0f; // Adjust this value as needed
            shader->set("minimumPickDistance", min_pick_distance);

            VAOBinder vao_bind(vao_);

            // Upload instance data - USE CACHED INSTANCES
            {
                BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
                upload_buffer(GL_ARRAY_BUFFER, std::span(cached_instances_), GL_DYNAMIC_DRAW);

                // Setup instance attributes
                // Instance transform matrix (locations 2-5, since 0=pos, 1=UV)
                for (int i = 0; i < 4; ++i) {
                    glEnableVertexAttribArray(2 + i);
                    glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                          reinterpret_cast<void*>(sizeof(glm::vec4) * i));
                    glVertexAttribDivisor(2 + i, 1);
                }

                // Instance color and alpha (location 6) - now vec4
                glEnableVertexAttribArray(6);
                glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                      reinterpret_cast<void*>(offsetof(InstanceData, color)));
                glVertexAttribDivisor(6, 1);

                // Instance texture ID (location 7) - needed even for picking, use IPointer for integer attribute
                glEnableVertexAttribArray(7);
                glVertexAttribIPointer(7, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                       reinterpret_cast<void*>(offsetof(InstanceData, texture_id)));
                glVertexAttribDivisor(7, 1);
                
                // Instance aspect ratio (location 8)
                glEnableVertexAttribArray(8);
                glVertexAttribPointer(8, 1, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                      reinterpret_cast<void*>(offsetof(InstanceData, aspect_ratio)));
                glVertexAttribDivisor(8, 1);
            }

            // Enable depth testing
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);

            // Draw solid faces only for picking
            {
                BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(face_ebo_);
                glDrawElementsInstanced(GL_TRIANGLES, num_face_indices_, GL_UNSIGNED_INT, 0, cached_instances_.size());
            }

            // Check for errors
            GLenum err = glGetError();
            if (err != GL_NO_ERROR) {
                LOG_ERROR("OpenGL error during picking render: 0x{:x}", err);
            }

            // Cleanup attributes
            for (int i = 2; i <= 8; ++i) {
                glDisableVertexAttribArray(i);
                if (i >= 2 && i <= 8) {
                    glVertexAttribDivisor(i, 0);
                }
            }
        }

        // Ensure rendering is complete before reading pixels
        glFinish();

        // Read pixel under mouse
        // Convert mouse position relative to viewport
        int pixel_x = static_cast<int>(mouse_pos.x - viewport_pos.x);
        int pixel_y = static_cast<int>(viewport_size.y - (mouse_pos.y - viewport_pos.y)); // Flip Y

        // Clamp to viewport bounds
        pixel_x = std::clamp(pixel_x, 0, picking_fbo_width_ - 1);
        pixel_y = std::clamp(pixel_y, 0, picking_fbo_height_ - 1);

        // Read a small area around the mouse position for debugging
        const int sample_size = 3;
        std::vector<float> pixels(sample_size * sample_size * 3);
        int read_x = std::max(0, pixel_x - 1);
        int read_y = std::max(0, pixel_y - 1);
        int read_width = std::min(sample_size, picking_fbo_width_ - read_x);
        int read_height = std::min(sample_size, picking_fbo_height_ - read_y);

        glReadPixels(read_x, read_y, read_width, read_height, GL_RGB, GL_FLOAT, pixels.data());

        // Get the center pixel (or first pixel if we're at the edge)
        int center_idx = 0;
        if (read_width == 3 && read_height == 3) {
            center_idx = 4 * 3; // Center of 3x3 is index 4
        } else if (read_width >= 2 && read_height >= 2) {
            // Try to get a center-ish pixel
            center_idx = ((read_height / 2) * read_width + (read_width / 2)) * 3;
        }

        float r = pixels[center_idx];
        float g = pixels[center_idx + 1];
        float b = pixels[center_idx + 2];

        // Decode ID from color
        int id = static_cast<int>(r * 255.0f + 0.5f) << 16 |
                 static_cast<int>(g * 255.0f + 0.5f) << 8 |
                 static_cast<int>(b * 255.0f + 0.5f);

        id -= 1; // We added 1 in the shader to avoid 0

        LOG_TRACE("Picked pixel at ({}, {}): RGB({:.3f}, {:.3f}, {:.3f}) -> ID {}",
                  pixel_x, pixel_y, r, g, b, id);

        // Restore previous FBO and viewport
        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
        glViewport(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);

        // Return camera ID if valid
        if (id >= 0 && id < static_cast<int>(camera_ids_.size())) {
            LOG_TRACE("Picked camera at index {} with ID {}", id, camera_ids_[id]);
            return camera_ids_[id];
        }

        return -1;
    }

} // namespace gs::rendering