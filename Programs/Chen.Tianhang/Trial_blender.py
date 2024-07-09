import bpy
import numpy as np
import math


class ViewOperator(bpy.types.Operator):
    bl_idname = "view3d.multimapping"
    bl_label = "Mapping_multiuv"

    def invoke(self, context, event):
        if context.space_data.type == 'VIEW_3D':
            for sel in bpy.context.selected_objects:
                bpy.context.scene.objects.active = bpy.data.objects[sel.name]
                bpy.ops.object.editmode_toggle()
                phi_inc = np.linspace(0, 2 * np.pi, 11)
                phi_roll = np.linspace(0, 2 * np.pi, 11)
                phi_inc = -phi_inc
                phi_inc.tolist()
                phi_roll.tolist()
                for iphi_inc in phi_inc:
                    for iphi_roll in phi_roll:
                        bpy.ops.transform.rotate(value=phi_inc,
                                                 orient_axis='X', orient_type='GLOBAL',
                                                 orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                                 orient_matrix_type='GLOBAL',
                                                 constraint_axis=(True, False, False),
                                                 mirror=True, use_proportional_edit=False,
                                                 proportional_edit_falloff='SMOOTH',
                                                 proportional_size=0.0182602,
                                                 use_proportional_connected=False,
                                                 use_proportional_projected=False, release_confirm=True)

                        bpy.ops.transform.rotate(value=iphi_roll,
                                                 orient_axis='Y', orient_type='GLOBAL',
                                                 orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                                 orient_matrix_type='GLOBAL',
                                                 constraint_axis=(False, True, False), mirror=True,
                                                 use_proportional_edit=False,
                                                 proportional_edit_falloff='SMOOTH',
                                                 proportional_size=0.0182602,
                                                 use_proportional_connected=False,
                                                 use_proportional_projected=False, release_confirm=True)

                        bpy.ops.uv.project_from_view(orthographic=False,
                                                     correct_aspect=True, clip_to_bounds=False, scale_to_bounds=False)
                        filepathRoot = 'D://Desktop/trial'
                        fileName = '%03d' % iphi_inc + '%03d' % iphi_roll + '.png'

                        bpy.context.area.ui_type = 'UV'
                        bpy.ops.uv.export_layout(filepath=filepathRoot + fileName,
                                                 size=(32768, 32768), opacity=0)
                        bpy.ops.object.editmode_toggle()

            return {'RUNNING_MODAL'}
        else:
            print('1')
            self.report({'WARNING'}, 'Active space must be a View3d')
            return {'CANCELLED'}


def menu_func(self, context):
    self.layout.operator(ViewOperator.bl_idname, text="Mapping_multiuv")


bpy.utils.register_class(ViewOperator)

# bpy.types.VIEW3D_OT_multimapping.append(menu_func)
bpy.context.area.ui_type = 'VIEW_3D'
bpy.ops.object.convert(target='MESH')
bpy.ops.object.select_all(action='SELECT')
bpy.context.space_data.shading.type = 'WIREFRAME'
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.view3d.multimapping('INVOKE_DEFAULT')