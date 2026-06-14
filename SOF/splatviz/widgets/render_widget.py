import torch
from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget
from arguments import SplattingSettings
from diff_gaussian_rasterization import SortMode, GlobalSortOrder, ExtendedSettings, DebugVisualization, DebugVisualizationType

class RenderWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Render")
        self.render_alpha = False
        self.render_depth = False
        self.render_normal = False
        self.render_distortion = False
        self.render_depth_normal = False
        self.render_depth_normal_loss = False
        self.image_informed_depthnormal = False
        self.manual_normalization = False
        self.use_mouse_pos = False
        self.scaling_modifier = 1
        self.depth_min_max = [0, 10000]
        self.resolution = 1024
        self.background_color = torch.tensor([0.0, 0.0, 0.0])
        self.splat_args = None
        self.debug_data = DebugVisualization()
        self.sh_degree = 3

    def _load_splat_args(self):
        ss = SplattingSettings(render=True)
        if 'ply_file_paths' in self.viz.args:
            self.splat_args = ss.get_settings_from_path(self.viz.args.ply_file_paths)
        else:
            self.splat_args = ExtendedSettings()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True, decoder=False):
        viz = self.viz
        if self.splat_args is None:
            self._load_splat_args()
        if show:
            label("Resolution", viz.label_w)
            _changed, self.resolution = imgui.input_int("##Resolution", self.resolution, 64)
            
            _, self.scaling_modifier = imgui.slider_float("Scaling Modifier", self.scaling_modifier, 0.001, 1.0)
            
            # max sh degree (to visualize diffuse)
            _, self.sh_degree = imgui.combo("Max SH Degree", self.sh_degree, ["0","1","2","3"])
            
            label("Background Color", viz.label_w)
            _changed, background_color = imgui.input_float3("##background_color", v=self.background_color.tolist(), format="%.1f")
            if _changed:
                self.background_color = torch.tensor(background_color)

            label("Render Alpha", viz.label_w)
            alpha_changed, self.render_alpha = imgui.checkbox("##RenderAlpha", self.render_alpha)

            label("Render Depth", viz.label_w)
            depth_changed, self.render_depth = imgui.checkbox("##RenderDepth", self.render_depth)
            
            if self.render_depth:
                imgui.indent()
                label("Manual Normalization", viz.label_w)
                _, self.manual_normalization = imgui.checkbox("##ManualNorm", self.manual_normalization)
                if self.manual_normalization:
                    _, self.depth_min_max = imgui.input_float2("Float2 Input", self.depth_min_max)
                imgui.unindent()
                
            label("Render Depth Normal", viz.label_w)
            depth_normal_changed, self.render_depth_normal = imgui.checkbox("##RenderDepthNormal", self.render_depth_normal)
            
            label("Render Normal", viz.label_w)
            normal_changed, self.render_normal = imgui.checkbox("##RenderNormal", self.render_normal)

            if self.render_alpha and alpha_changed:
                self.render_depth = False
                self.render_normal = False
                self.render_depth_normal = False
            if self.render_depth and depth_changed:
                self.render_alpha = False
                self.render_normal = False
                self.render_depth_normal = False
            if self.render_normal and normal_changed:
                self.render_depth = False
                self.render_alpha = False
                self.render_depth_normal = False
            if self.render_depth_normal and depth_normal_changed:
                self.render_depth = False
                self.render_alpha = False
                self.render_normal = False
                
            if imgui.collapsing_header("Render Losses"):
                label("Render Distortion", viz.label_w)
                distortion_changed, self.render_distortion = imgui.checkbox("##RenderDistortion", self.render_distortion)
                
                label("Render Depth/Normal Loss", viz.label_w)
                depth_normal_loss_changed, self.render_depth_normal_loss = imgui.checkbox("##RenderDepthNormalLoss", self.render_depth_normal_loss)
                if self.render_depth_normal_loss:
                    imgui.indent()
                    label("Image Informed", viz.label_w)
                    _, self.image_informed_depthnormal = imgui.checkbox("##ImageInformed", self.image_informed_depthnormal)
                    imgui.unindent()
                
                if distortion_changed and self.render_distortion:
                    self.render_depth_normal_loss = False
                if depth_normal_loss_changed and self.render_depth_normal_loss:
                    self.render_distortion = False
                
            def emitCheckbox(obj, attr_name):
                label(attr_name, viz.label_w)
                
                value = getattr(obj, attr_name)  # Get the attribute value
                changed, new_value = imgui.checkbox(f"##{attr_name}", value)
                
                if changed:
                    setattr(obj, attr_name, new_value) 
                
            if imgui.collapsing_header("Splatting Settings"):
                        
                emitCheckbox(self.splat_args, "exact_depth")
                emitCheckbox(self.splat_args, "proper_ewa_scaling")
                emitCheckbox(self.splat_args, "load_balancing")
                # if imgui.collapsing_header("Sort Settings"):
                _, self.splat_args.sort_settings.sort_order = imgui.combo("Sort Order", self.splat_args.sort_settings.sort_order, 
                    [str(x) for x in [GlobalSortOrder.Z_DEPTH, GlobalSortOrder.DISTANCE, GlobalSortOrder.PTD_CENTER, GlobalSortOrder.PTD_MAX]])
                _, self.splat_args.sort_settings.sort_mode = imgui.combo("Sort Mode", self.splat_args.sort_settings.sort_mode, 
                    [str(x) for x in [SortMode.GLOBAL, SortMode.PPX_FULL, SortMode.PPX_KBUFFER, SortMode.HIER]])
                    # _, self.splat_args.sort_settings.sort_mode = imgui.combo("Sort Mode", self.splat_args.sort_settings.sort_mode, 
                    #     [str(x) for x in [SortMode.GLOBAL, SortMode.HIER]])  
                if imgui.collapsing_header("Culling Settings"):
                    emitCheckbox(self.splat_args.culling_settings, "rect_bounding")
                    emitCheckbox(self.splat_args.culling_settings, "tight_opacity_bounding")
                    emitCheckbox(self.splat_args.culling_settings, "tile_based_culling")
                    emitCheckbox(self.splat_args.culling_settings, "hierarchical_4x4_culling")
                    
            if imgui.collapsing_header("Debug Visualization"):
                _, self.debug_data.type = imgui.combo("Debug Visualization Type", self.debug_data.type, 
                    [str(x) for x in list(DebugVisualizationType)])
                imgui.separator()
                
                if self.debug_data.type is not DebugVisualizationType.DISABLED.value:
                    # per pixel debugging
                    _xy_changed, xy_new = imgui.input_int2("Debug Pixel", v=[self.debug_data.debugX, self.debug_data.debugY])
                    if _xy_changed:
                        self.debug_data.debugX = xy_new[0]
                        self.debug_data.debugY = xy_new[1]
                    _, self.use_mouse_pos = imgui.checkbox('Use Mouse Position', self.use_mouse_pos)
                    
                    if self.use_mouse_pos:
                        # we need to let everyone know the mouse is clicked
                        imgui.indent()
                        imgui.text("Click on the texture using the right mouse button to select a pixel!")
                        imgui.unindent()
                        if imgui.is_mouse_clicked(1):
                            mouse_pos = imgui.get_mouse_pos()
                            self.debug_data.debugX = int((mouse_pos[0] - viz.args.tex_pos[0]) / viz.args.zoom)
                            self.debug_data.debugY = int((mouse_pos[1] - viz.args.tex_pos[1]) / viz.args.zoom)
                    
                    # debug normalization
                    emitCheckbox(self.debug_data, "debug_normalize")
                    if self.debug_data.debug_normalize:
                        imgui.indent()
                        _minmax_changed, minmax_new = imgui.input_float2("Min Max", v=[self.debug_data.min, self.debug_data.max], format="%.3f")
                        if _minmax_changed:
                            self.debug_data.min = minmax_new[0]
                            self.debug_data.max = minmax_new[1]
                        imgui.unindent()
                        
                    # precision
                    _, self.debug_data.precision = imgui.input_int("Precision", self.debug_data.precision)
                    
                    # timing
                    emitCheckbox(self.debug_data, "timing_enabled")

        viz.args.background_color = self.background_color
        viz.args.resolution = self.resolution
        viz.args.render_alpha = self.render_alpha
        viz.args.render_depth = self.render_depth
        viz.args.render_normal = self.render_normal
        viz.args.render_distortion = self.render_distortion
        viz.args.render_depth_normal = self.render_depth_normal
        viz.args.render_depth_normal_loss = self.render_depth_normal_loss
        viz.args.image_informed_depthnormal = self.image_informed_depthnormal
        viz.args.splat_args = self.splat_args
        viz.args.debug_data = self.debug_data
        viz.args.depth_min_max = self.depth_min_max
        viz.args.manual_normalization = self.manual_normalization
        viz.args.scaling_modifier = self.scaling_modifier
        viz.args.sh_degree = self.sh_degree
