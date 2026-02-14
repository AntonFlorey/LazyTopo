import bpy
import gpu
from gpu_extras.batch import batch_for_shader

def deactivate_draw_callback(callback_handle, region_type='WINDOW'):
    if callback_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(callback_handle, region_type)

def redraw_view_3d(context : bpy.types.Context):
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

def textured_triangles_draw_callback(vertex_positions, triangle_indices, uvs, texture_image_path):
    # load the texture image
    image = bpy.data.images.load(filepath=texture_image_path, check_existing=True)
    texture = gpu.texture.from_image(image)
    shader = gpu.shader.from_builtin('IMAGE')
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertex_positions, "texCoord": uvs}, indices=triangle_indices)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    shader.bind()
    shader.uniform_sampler("image", texture)
    batch.draw(shader)
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('NONE')
    
def uniform_lines_2D_draw_callback(lines_batch: gpu.types.GPUBatch, color, width=3):
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    prev_line_width = gpu.state.line_width_get()
    gpu.state.line_width_set(width)
    shader.bind()   
    shader.uniform_float("color", color)
    lines_batch.draw(shader)
    gpu.state.line_width_set(prev_line_width)

def smooth_color_lines_draw_callback(colored_lines_batch: gpu.types.GPUBatch, width=3.0):
    shader = gpu.shader.from_builtin("SMOOTH_COLOR")
    prev_line_width = gpu.state.line_width_get()
    gpu.state.line_width_set(width)
    shader.bind()   
    colored_lines_batch.draw(shader)
    gpu.state.line_width_set(prev_line_width)
