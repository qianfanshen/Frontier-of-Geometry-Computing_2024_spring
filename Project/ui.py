import open3d as o3d
from open3d.visualization import gui, rendering
import numpy as np
import copy
from utils.util import *
from func.simplification_model import generate_model
from func.smoothing_model import MeshSmoothing
from func.simplification_model import simplification
from func.tutte_embedding_model import TutteEmbedding
from func.arap_model import *

class App:
    MENU_OPEN = 1

    MENU_SHOW = 5

    MENU_QUIT = 20
    MENU_ABOUT = 21

    show = True

    _picked_indicates = []
    _picked_points = []
    _pick_num = 0

    x_ = None
    y_ = None

    mode = "Deformation"
    shader = "defaultLit"

    fixed = []
    selected = []       # 变换点云
    handle = []
    handle_idx = []   # handle的标号
    handle_point = None  # handle点的坐标
    file_path = None

    _area_size = 0
    _iter = 0
    scale = 0
    _d_iteration = 0

    method = "explicit"
    lr = 1e-2
    out_path = "output.obj"
    ratio = 0.1
    depth_image = None
    added_line = False

    _label3d_list = []

    def __init__(self):
        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window("FGC Final Project", 1600, 1200)
        w = self.window
        em = w.theme.font_size

        # 渲染窗口
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_mouse(self._on_mouse_widget3d)

        self._info = gui.Label("")
        self._info.visible = False

        self.flag = False
        self.mesh = None
        self.pcd = None

        # 右侧面板
        self._pannel = gui.CollapsableVert('Settings', 0, gui.Margins(0.25*em, 0.25*em, 0.25*em, 0.25*em))

        self._mode = gui.Combobox()
        self._mode.add_item("Deformation")
        self._mode.add_item("Smoothing")
        self._mode.add_item("Simplify")
        self._mode.add_item("Parameterization")
        self._mode.add_item("Check")
        self._mode.set_on_selection_changed(self._mode_changed)
        self._mode_pannel = gui.Horiz()
        self._mode_pannel.add_child(gui.Label("Mode"))
        self._mode_pannel.add_child(self._mode)
        self._pannel.add_child(self._mode_pannel)
        self._pannel.add_fixed(0.25 * em)

        self._shader = gui.Combobox()
        self._shader.add_item("defaultLit")
        self._shader.add_item("defaultUnlit")
        self._shader.add_item("normals")
        self._shader.add_item("unlitLine")
        self._shader.set_on_selection_changed(self._shader_changed)
        self._shader_pannel = gui.Horiz()
        self._shader_pannel.add_child(gui.Label("Shader"))
        self._shader_pannel.add_child(self._shader)
        self._pannel.add_child(self._shader_pannel)
        self._pannel.add_fixed(0.25 * em)

        self.output_path = gui.TextEdit()
        self.output_path.set_on_value_changed(self.change_outpath)
        self._pannel.add_child(gui.Label("Output Filename (default: output)"))
        self._pannel.add_child(self.output_path)
        self._pannel.add_fixed(0.25 * em)

        # Deformation
        self.area_size = gui.Slider(gui.Slider.Type(0))
        self.area_size.set_limits(0, 500)
        self.area_size.set_on_value_changed(self._area_size_changed)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Nearest Points Num"))
        grid.add_child(self.area_size)

        self._iteration = gui.Slider(gui.Slider.Type(0))
        self._iteration.set_limits(0, 50)
        self._iteration.set_on_value_changed(self.d_iteration_changed)
        _iter_grid = gui.VGrid(2, 0.25 * em)
        _iter_grid.add_child(gui.Label("Iteration"))
        _iter_grid.add_child(self._iteration)

        self._deformation = gui.CollapsableVert('Deformation', 1, gui.Margins(0.1*em, 0.1*em, 0.1*em, 0.1*em))
        self._deformation.add_child(grid)
        self._deformation.add_child(_iter_grid)
        self._pannel.add_child(self._deformation)

        # Smoothing
        self.smoothing = gui.CollapsableVert('Smoothing', 1, gui.Margins(0.1*em, 0.1*em, 0.1*em, 0.1*em))

        self.iteration = gui.Slider(gui.Slider.Type(0))
        self.iteration.set_limits(0, 50)
        self.iteration.set_on_value_changed(self._iteration_changed)
        iter_grid = gui.VGrid(2, 0.25 * em)
        iter_grid.add_child(gui.Label("Iteration"))
        iter_grid.add_child(self.iteration)

        self.smoothing_method = gui.Combobox()
        self.smoothing_method.add_item("explicit")
        self.smoothing_method.add_item("implicit")
        self.smoothing_method.set_on_selection_changed(self._smoothing_method_changed)

        self.learning_rate = gui.Slider(gui.Slider.Type(1))
        self.learning_rate.set_limits(0, 1)
        self.learning_rate.set_on_value_changed(self._lr_changed)
        lr_grid = gui.VGrid(2, 0.25 * em)
        lr_grid.add_child(gui.Label("Learning Rate"))
        lr_grid.add_child(self.learning_rate)

        self.smoothing_layout = gui.Horiz()
        self.smoothing_layout.add_child(gui.Label("Method"))
        self.smoothing_layout.add_fixed(0.15 * em)
        self.smoothing_layout.add_child(self.smoothing_method)
        self.smoothing.add_child(self.smoothing_layout)
        self.smoothing.add_fixed(0.15 * em)
        self.smoothing.add_child(iter_grid)
        self.smoothing.add_fixed(0.15 * em)
        self.smoothing.add_child(lr_grid)
        self.smoothing.add_fixed(0.15 * em)
        self._pannel.add_child(self.smoothing)
        self._pannel.add_fixed(0.25 * em)

        # Simplify
        self.simplify = gui.CollapsableVert('Simplify', 1, gui.Margins(0.1*em, 0.1*em, 0.1*em, 0.1*em))
        self._ratio = gui.Slider(gui.Slider.Type(1))
        self._ratio.set_limits(0, 1)
        self._ratio.set_on_value_changed(self._ratio_changed)
        ratio_grid = gui.VGrid(2, 0.25 * em)
        ratio_grid.add_child(gui.Label("Ratio"))
        ratio_grid.add_child(self._ratio)
        self.simplify.add_child(ratio_grid)
        self._pannel.add_child(self.simplify)

        # Parameterization
        self._parameterization = gui.CollapsableVert('Parameterization', 1, gui.Margins(0.1*em, 0.1*em, 0.1*em, 0.1*em))
        self._pannel.add_child(self._parameterization)


        self._button = gui.Button("Reset Mesh")
        self._button.set_on_clicked(self._reset_mesh)

        # 布局回调函数
        w.set_on_layout(self._on_layout)
        w.add_child(self._info)
        w.add_child(self._scene)
        w.add_child(self._pannel)
        w.add_child(self._button)

        # ---------------Menu----------------
        if gui.Application.instance.menubar is None:
            # 退出界面
            quit_menu = gui.Menu()
            quit_menu.add_item("Quit", App.MENU_QUIT)

            # 文件菜单栏
            file_menu = gui.Menu()
            file_menu.add_item("Open", App.MENU_OPEN)

            # 显示菜单栏
            show_menu = gui.Menu()
            show_menu.add_item("Show Geometry", App.MENU_SHOW)
            show_menu.set_checked(App.MENU_SHOW, True)

            # 帮助菜单栏
            help_menu = gui.Menu()
            help_menu.add_item("About", App.MENU_ABOUT)
            help_menu.set_enabled(App.MENU_ABOUT, True)

            # 菜单栏
            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Show", show_menu)
            menu.add_menu("Help", help_menu)
            menu.add_menu("Quit", quit_menu)

            gui.Application.instance.menubar = menu
            
            #-----注册菜单栏事件------
            w.set_on_menu_item_activated(App.MENU_OPEN, self._menu_open)
            w.set_on_menu_item_activated(App.MENU_QUIT, self._menu_quit)
            w.set_on_menu_item_activated(App.MENU_SHOW, self._menu_show)
            w.set_on_menu_item_activated(App.MENU_ABOUT, self._menu_about)

    def _reset_mesh(self):
        if self.flag:
            mesh = o3d.io.read_triangle_mesh(self.file_path)
            mesh.compute_vertex_normals()
            self.mesh = mesh
            material = rendering.MaterialRecord()
            material.shader = self.shader
            if self.added_line:
                self._scene.scene.remove_geometry('Lines')
                self.added_line = False
            self._scene.scene.remove_geometry('Mesh Deformation')
            self._scene.scene.add_geometry('Mesh Deformation', self.mesh, material)
            self._scene.force_redraw()

    def _shader_changed(self, shader, index):
        self.shader = shader
        material = rendering.MaterialRecord()
        material.shader = self.shader
        self._scene.scene.remove_geometry('Mesh Deformation')
        self._scene.scene.add_geometry('Mesh Deformation', self.mesh, material)
        self._scene.force_redraw()

    def _mode_changed(self, mode, index):
        self.mode = mode

    def _ratio_changed(self, ratio):
        self.ratio = ratio

    def change_outpath(self, text):
        self.out_path = text + ".obj"
        print(self.out_path)
        
    def _smoothing_method_changed(self, name, index):
        print(index)
        self.method = name

    def _lr_changed(self, value):
        self.lr = value

    # 鼠标事件
    def _on_mouse_widget3d(self, event):
        if self.mode == "Deformation" and self.flag:
            if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT) and event.is_modifier_down(gui.KeyModifier.CTRL):
                def depth_callback(depth_image):
                    x = event.x - self._scene.frame.x
                    y = event.y - self._scene.frame.y
                    self.depth_image = depth_image
                    depth = np.asarray(depth_image)[y, x]

                    if self.area_size != 0:
                        if depth==1.0:
                            # 远平面（没有几何体）
                            text = ""
                        else:
                            world = self._scene.scene.camera.unproject(x, y, depth, self._scene.frame.width, self._scene.frame.height)

                            text = "({:.3f}, {:.3f}, {:.3f})".format(world[0],world[1],world[2])

                            idx = self._cacl_prefer_indicate(world)
                            true_point = np.asarray(self.pcd.points)[idx]
                            # s = self.area_size
                            # size = np.array([s, s, s])
                            # min_bound = true_point - 0.5 * size
                            # max_bound = true_point + 0.5 * size

                            # crop = o3d.geometry.AxisAlignedBoundingBox(min_bound.tolist(), max_bound.tolist())
                            # handle_points = self.pcd.crop(crop).paint_uniform_color([1, 0, 0])
                            kdt = o3d.geometry.KDTreeFlann(self.pcd)
                            [_, indices, _] = kdt.search_knn_vector_3d(true_point, self._area_size)
                            self.handle_idx.append(indices)
                            [_, f_indices, _] = kdt.search_knn_vector_3d(true_point, math.ceil(0.1 * self._area_size))
                            self.fixed.append(f_indices)
                            selected_points = self.pcd.select_by_index(indices).paint_uniform_color([1, 0, 0])
                            handle_points = self.pcd.select_by_index(f_indices).paint_uniform_color([0, 1, 0])

                            self._pick_num += 1
                            self._picked_indicates.append(idx)
                            self._picked_points.append(true_point)
                            self.selected.append(selected_points)
                            self.handle.append(handle_points)
                            self.handle_point = true_point
                            
                            print(f"Pick point #{idx} at ({true_point[0]}, {true_point[1]}, {true_point[2]})")
                    
                        def draw_point():
                            self._info.text = text
                            self._info.visible = (text != "")
                            self.window.set_needs_layout()

                            if depth != 1.0:
                                label3d = self._scene.add_3d_label(true_point, "#"+str(self._pick_num))
                                self._label3d_list.append(label3d)

                                # 标记球
                                sphere = self.selected[-1]
                                handle_point = self.handle[-1]
                                material = rendering.MaterialRecord()
                                material.shader = self.shader
                                self._scene.scene.add_geometry("handle"+str(self._pick_num), sphere, material)
                                self._scene.scene.add_geometry('handle_point'+str(self._pick_num), handle_point, material)
                                self._scene.force_redraw()
                        
                        gui.Application.instance.post_to_main_thread(self.window, draw_point)
                
                self._scene.scene.scene.render_to_depth_image(depth_callback)
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.RIGHT) and event.is_modifier_down(gui.KeyModifier.CTRL):
                if self._pick_num > 0:
                    idx = self._picked_indicates.pop()
                    point = self._picked_points.pop()
                    self.handle.pop()
                    self.selected.pop()
                    self.handle_idx.pop()
                    self.fixed.pop()
                    
                    print(f"Undo pick: #{idx} at ({point[0]}, {point[1]}, {point[2]})")
                    
                    self._scene.scene.remove_geometry('handle'+str(self._pick_num))
                    self._scene.scene.remove_geometry('handle_point'+str(self._pick_num))
                    self._pick_num -= 1
                    self._scene.remove_3d_label(self._label3d_list.pop())
                    self._scene.force_redraw()
                else:
                    print("Undo no point!")
                return gui.Widget.EventCallbackResult.HANDLED
        
            elif event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.SHIFT):
                self.x_ = self.handle_point[0]
                self.y_ = self.handle_point[1]
                self.z_ = self.handle_point[2]
                return gui.Widget.EventCallbackResult.CONSUMED
            
            elif event.type == gui.MouseEvent.Type.BUTTON_UP and event.is_modifier_down(gui.KeyModifier.SHIFT):
                def depth_callback(depth_image):
                    x = (event.x - self._scene.frame.x)
                    y = (event.y - self._scene.frame.y)
                    depth = np.asarray(depth_image)[y, x]
                    if depth != 1.0:
                        world = self._scene.scene.camera.unproject(x, y, depth, self._scene.frame.width, self._scene.frame.height)
                        point = np.array([self.x_, self.y_, self.z_])
                        point_w = np.array([world[0], world[1], world[2]])
                        delta_x = (world[0] - self.x_)
                        delta_y = (world[1] - self.y_)
                        delta_z = (world[2] - self.z_)
                        delta_pos = np.array([delta_x, delta_y, delta_z])
                        rotation_angle = np.arctan2(np.linalg.norm(delta_pos), np.dot(point_w, point))
                        print(delta_pos)
                        
                        lines = [[0, 1]]
                        triangle_points = np.array([[world[0], world[1], world[2]], [self.x_, self.y_, self.z_]])
                        color = np.array([[1, 0, 0]])
                        line_set = o3d.geometry.LineSet()
                        line_set.lines = o3d.utility.Vector2iVector(lines) 
                        line_set.colors = o3d.utility.Vector3dVector(color)
                        line_set.points = o3d.utility.Vector3dVector(triangle_points)
                        material = rendering.MaterialRecord()
                        material.shader = self.shader
                        if self.added_line:
                            self._scene.scene.remove_geometry('Lines')
                        self._scene.scene.add_geometry('Lines', line_set, material)
                        self.added_line = True

                        print("Deformation")
                        deformed_mesh = deformation(self.file_path, self._d_iteration, self.handle[-1], self.handle_idx[-1], self.fixed[-1], delta_pos, self.out_path, rotation_angle=rotation_angle)
                        mesh_ = o3d.io.read_triangle_mesh(self.out_path)
                        mesh_.compute_vertex_normals()
                        self.mesh = mesh_
                        material = rendering.MaterialRecord()
                        material.shader = self.shader
                        
                        self._scene.scene.remove_geometry('Mesh Deformation')
                        self._scene.scene.add_geometry('Mesh Deformation', mesh_, material)
                        # self._scene.scene.remove_geometry('Lines')
                    else:
                        print("WARNING: Drag Again")
                self._scene.scene.scene.render_to_depth_image(depth_callback)
                return gui.Widget.EventCallbackResult.CONSUMED

        elif self.mode == "Smoothing" and self.flag and event.is_modifier_down(gui.KeyModifier.CTRL) and event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            smoother = MeshSmoothing(self.file_path)
            smoother.smooth(self.iteration.int_value, self.out_path, self.method, self.lr)
            print(f"Model smoothed with iterations {self.iteration.int_value} and method {self.method} and lr {self.lr}")
            material = rendering.MaterialRecord()
            material.shader = self.shader
            print(self.out_path)
            mesh_ = o3d.io.read_triangle_mesh(self.out_path)
            mesh_.compute_vertex_normals()
            self.mesh = mesh_
            self._scene.scene.remove_geometry('Mesh Deformation')
            self._scene.scene.add_geometry('Mesh Deformation', mesh_, material)
            return gui.Widget.EventCallbackResult.HANDLED
        
        elif self.mode == "Simplify" and self.flag and event.is_modifier_down(gui.KeyModifier.CTRL) and event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            model = simplification(self.file_path, 0, self.ratio)
            model.generate_valid_pairs()
            model.calculate_optimal_contraction_pairs_and_cost()
            model.iteratively_remove_pairs()
            model.generate_new_3d_model()
            model.output(self.out_path)
            print(f"Model simplified with ratio {self.ratio}")
            material = rendering.MaterialRecord()
            material.shader = self.shader
            mesh_ = o3d.io.read_triangle_mesh(self.out_path)
            mesh_.compute_vertex_normals()
            self.mesh = mesh_
            self._scene.scene.remove_geometry('Mesh Deformation')
            self._scene.scene.add_geometry('Mesh Deformation', mesh_, material)
            return gui.Widget.EventCallbackResult.HANDLED
        
        elif self.mode =="Parameterization" and self.flag and event.is_modifier_down(gui.KeyModifier.CTRL) and event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            tutte = TutteEmbedding(self.file_path)
            tutte.parameterize(self.out_path)
            tutte.color_mesh()
            tutte.write_mesh(self.out_path)
            material = rendering.MaterialRecord()
            material.shader = self.shader
            mesh_ = o3d.io.read_triangle_mesh(self.out_path)
            mesh_.compute_vertex_normals()
            self.mesh = mesh_
            self._scene.scene.remove_geometry('Mesh Deformation')
            self._scene.scene.add_geometry('Mesh Deformation', mesh_, material)
            return gui.Widget.EventCallbackResult.HANDLED

        elif self._mode == "Check":
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED


    def _cacl_prefer_indicate(self, point):
        pcd = copy.deepcopy(self.pcd)
        pcd.points.append(np.asarray(point))

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        [k, idx, _]=pcd_tree.search_knn_vector_3d(pcd.points[-1], 2)
        return idx[-1]

    # 打开并显示一个obj模型
    def _menu_open(self):
        # 文件拾取对话框
        file_picker = gui.FileDialog(gui.FileDialog.OPEN, "Select file...", self.window.theme)
        
        # 文件类型过滤
        file_picker.add_filter('.obj', 'obj model files')
        file_picker.add_filter('', 'All files')
        
        # 初始文件路径
        file_picker.set_path('./models')

        # 设置对话框按钮回调
        file_picker.set_on_cancel(self._on_cancel)
        file_picker.set_on_done(self._on_done)

        # 显示对话框
        self.window.show_dialog(file_picker)
    
    def _on_cancel(self):
        # 关闭当前对话框
        self.window.close_dialog()

    def _on_done(self, filename): 
        self.window.close_dialog()
        self.load(filename)

    def load(self, file):
        # 读取模型文件
        mesh = o3d.io.read_triangle_mesh(file)
        mesh.compute_vertex_normals()
        self.mesh = mesh
        self.file_path = file
        # self.mesh_d = Mesh.load_from_obj(self.file_path)
        # self.mesh_s = load_obj(self.file_path)

        pcd = o3d.geometry.PointCloud()
        self.pcd = pcd
        self.pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        self.pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))

        # 定义材质
        material = rendering.MaterialRecord()
        material.shader = 'defaultLit'
        
        # 向场景中添加模型
        if not self.flag:
            print("New Scene")
            self._scene.scene.add_geometry('Mesh Deformation', mesh, material)
            self.flag = True
        else:
            while self._pick_num > 0:
                self._picked_indicates.pop()
                self._picked_points.pop()

                self._scene.scene.remove_geometry("handle"+str(self._pick_num))
                self._scene.scene.remove_geometry("handle_point"+str(self._pick_num))
                self.bounding_box.pop()
                self.handle.pop()
                self._pick_num -= 1
                self._scene.remove_3d_label(self._label3d_list.pop())
            self._scene.scene.remove_geometry('Mesh Deformation')
            self._scene.scene.add_geometry('Mesh Deformation', mesh, material)
        bounds = mesh.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds,bounds.get_center())

        # 重绘
        self._scene.force_redraw()

    # 退出应用
    def _menu_quit(self):
        self.window.close()

    # 切换显示模型
    def _menu_show(self):
        self.show = not self.show
        gui.Application.instance.menubar.set_checked(App.MENU_SHOW, self.show)
        self._scene.scene.show_geometry('Mesh Deformation', self.show)

    def _menu_about(self):
        self.window.show_message_box('Push Info', 
                                    "Usage Method: \n   Menu: \nClick on the File menu at the top left corner to select the Mesh model (.obj file) to be used.\nClick the Reset Mesh button at the bottom left corner to restore the mesh.\n   Settings: \nIn the Setting panel on the right side, select Mode to choose from Mesh Deformation, Smoothing, Simplification, Parameterization, and Check modes. The Check mode is used for observing the 3D object after operations and does not perform any operations on the mesh.\nIn the Setting panel on the right side, select Shader to apply coloring to the Mesh. There are four types of shaders to choose from: defaultLit, defaultUnlit, normals, and unlitLine.\nFill in the Output Filename field with the name of the processed mesh file to be exported. All exported meshes are in .obj format and are located in the (./models) directory, with the default name (output.obj). Only the file name for export needs to be filled in this field (for example, for output.obj, just write output).\n   Deformation:\nSlide the Nearest Points Num to select the nearest x points around the handle point as the input point cloud for the Deformation algorithm.Slide the Iteration to choose the number of deformation iterations.\nHold the CTRL key and use the left mouse button to select the handle point.\nHold the CTRL key and use the right mouse button to deselect the handle point.\nHold the SHIFT key and use the left mouse button to drag and apply a specific deformation direction to the given mesh (note that if you slide to the far plane, a WARNING will be output).\n   Smoothing:\nYou can choose between explicit or implicit methods for mesh smoothing. \nSlide the Iteration slider to select the number of smoothing iterations. \nSlide the Learning Rate slider to select the rate of change for smoothing. \nHold the CTRL key and use the left mouse button to start Smoothing. \n   Simplification: \nSlide the Ratio slider to select the proportion of remaining vertices after simplification relative to the original number of vertices. \nHold the CTRL key and use the left mouse button to start Simplification.\n   Parameterization: \nHold the CTRL key and use the left mouse button to start Parameterization. \n   Check:\nHold the left mouse button to drag and rotate the object. \nHold the CTRL key and the left mouse button to drag and change the object's position. \nHold the SHIFT key and scroll the mouse wheel to zoom in and out.")

    def _area_size_changed(self, size):
        self._area_size = int(size)

    def _iteration_changed(self, size):
        self._iter = int(size)

    def d_iteration_changed(self, size):
        self._d_iteration = int(size)

    def scale_changed(self, size):
        self.scale= 10**(-size)

    def _on_layout(self, layout_context):
        #   在on_layout回调函数中应正确设置所有子对象的框架(position + size)，
        #   回调结束之后才会布局孙子对象。
        r = self.window.content_rect
        self._scene.frame = r

        pref = self._info.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self._info.frame = gui.Rect(r.get_right() - pref.width, r.get_bottom() - pref.height, pref.width, pref.height)
        
        r = self.window.content_rect
        self._scene.frame = r

        pannel_width = 17 * layout_context.theme.font_size
        pannel_height = min(
            r.height, self._pannel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self._pannel.frame = gui.Rect(r.get_right() - pannel_width, r.y, pannel_width,pannel_height)

        button_pref = self._button.calc_preferred_size(
            layout_context, gui.Widget.Constraints())
        self._button.frame = gui.Rect(r.x, r.get_bottom() - button_pref.height, button_pref.width, button_pref.height)

    def run(self):
        gui.Application.instance.run()