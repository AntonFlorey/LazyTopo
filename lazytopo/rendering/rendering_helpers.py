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
    