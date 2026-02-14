import bpy

def trigger_lazytopo_panels_redraw(context: bpy.types.Context):
    if context.area is None or context.area.regions is None:
        return
    for region in context.area.regions:
        if region.active_panel_category == "LazyTopo":
            region.tag_redraw()

def write_custom_split_property_row(layout : bpy.types.UILayout, text, data, prop_name, split_factor, active=True):
    custom_row = layout.row().split(factor=split_factor, align=True)
    col_1, col_2 = (custom_row.column(), custom_row.column())
    col_1.label(text=text)
    col_2.prop(data, prop_name, text="")
    custom_row.active = active
    