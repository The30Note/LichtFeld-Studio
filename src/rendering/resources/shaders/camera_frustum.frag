#version 430 core

// Inputs from vertex shader
in vec3 FragPos;
in vec4 vertexColor;
in vec2 TexCoord;
flat in int instanceID;
flat in uint textureID;

// Output
out vec4 FragColor;

// Uniforms
uniform vec3 viewPos;
uniform int highlightIndex = -1;
uniform vec3 highlightColor = vec3(1.0, 0.85, 0.0);
uniform bool pickingMode = false;
uniform float minimumPickDistance = 0.5;
uniform bool showImages = false;
uniform float imageOpacity = 0.25;
uniform sampler2D cameraTexture;
uniform vec4 wireColor = vec4(1.0, 1.0, 1.0, 1.0);
uniform bool wireframeMode = false;

void main() {
    if (pickingMode) {
        float distance = length(viewPos - FragPos);
        if (distance < minimumPickDistance) {
            discard;
        }
        FragColor = vertexColor;
        return;
    }

    if (showImages && textureID > 0u) {
        vec4 imageColor = texture(cameraTexture, TexCoord);
        vec4 finalColor = vec4(imageColor.rgb, imageOpacity * imageColor.a);
        
        if (finalColor.a < 0.01) {
            discard;
        }
        
        FragColor = finalColor;
        return;
    }

    // Use wire color for wireframe rendering, vertex color for solid faces
    vec4 finalColor = wireframeMode ? wireColor : vertexColor;
    
    // Apply highlight only to solid faces, not wireframes
    if (instanceID == highlightIndex) {
        finalColor.rgb = highlightColor;
        finalColor.a = min(1.0, finalColor.a + 0.3);
    }

    FragColor = finalColor;
}
