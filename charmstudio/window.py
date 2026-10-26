from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSettings
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QLineEdit, QTreeWidget, QTreeWidgetItem, QComboBox,
    QDoubleSpinBox, QSlider, QCheckBox, QGroupBox, QFormLayout, QPlainTextEdit,
    QFileDialog, QMessageBox, QTabWidget, QScrollArea, QToolBar, QSpinBox,
)
from .resources import Resources
from .source import Model
from .viewport import Viewport
from .glviewport import GLViewport
from .controls import TransformControl
from .charm_pose import reference_vertices
from .project import Placement, digest, charm_id
from .vpk import write_vpk
from .addons import package_charm


STYLE = '''
QWidget { background: #17202d; color: #dce6f3; font: 10pt "Segoe UI"; }
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #111923; }
QToolBar { background: #1c2938; border: none; spacing: 8px; padding: 8px; }
QToolBar QLabel { background: transparent; }
QPushButton { background: #293b50; border: 1px solid #3a506a; border-radius: 5px; padding: 7px 11px; }
QPushButton:hover { background: #354c66; }
QPushButton:disabled { color: #65748a; background: #1e2b3a; }
QPushButton#primary { color: #102321; background: #66d4c5; border: none; font-weight: 600; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit { background: #111a26; border: 1px solid #35455c; border-radius: 4px; padding: 5px; }
QTreeWidget { background: #131d29; border: none; alternate-background-color: #182535; }
QTreeWidget::item { padding: 4px 1px; }
QTreeWidget::item:selected { background: #285a61; }
QGroupBox { border: 1px solid #304056; border-radius: 6px; margin-top: 12px; padding: 12px 9px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 5px; }
QTabWidget::pane { border: 1px solid #304056; }
QTabBar::tab { background: #202e40; padding: 8px 18px; }
QTabBar::tab:selected { background: #31536a; }
QSlider::groove:horizontal { height: 5px; background: #314459; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; background: #6bd5c7; margin: -4px 0; border-radius: 5px; }
QStatusBar { background: #101822; color: #a3b4c9; }
QLabel#subtle { color: #9daec4; }
'''


class MainWindow(QMainWindow):
    jobFinished = pyqtSignal(object)

    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.resources = Resources()
        self.weapon = self.charm = None
        self.weapon_mount = self.charm_mount = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobFinished.connect(self.finish_job)
        self.loading = 0
        self.settings = QSettings()
        self.frame = 0.
        self.last_tick = time.monotonic()
        self.setWindowTitle('L4D2 Charm Studio 1.5.1 · Addon natural experimental')
        self.resize(1450, 900)
        self.setStyleSheet(STYLE)
        self.build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(16 if isinstance(self.viewport, GLViewport) else 33)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        QTimer.singleShot(0, self.load_defaults)

    def build_ui(self):
        toolbar = QToolBar('Contenido')
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        brand = QLabel('  CHARM STUDIO   /   L4D2   ')
        brand.setStyleSheet('color: #73dfd0; font-weight: 700; font-size: 13pt;')
        toolbar.addWidget(brand)
        for label, callback in [('Abrir VPK', self.open_vpk), ('Abrir carpeta addon', self.open_folder),
                                ('Cargar juego', self.open_game), ('Abrir perfil', self.load_project)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        splitter = QSplitter()
        self.setCentralWidget(splitter)
        left = QWidget()
        left.setMinimumWidth(220)
        layout = QVBoxLayout(left)
        layout.addWidget(QLabel('EXPLORADOR DE CONTENIDO'))
        self.search = QLineEdit()
        self.search.setPlaceholderText('Filtrar rutas…')
        self.search.textChanged.connect(self.filter_tree)
        layout.addWidget(self.search)
        self.only_models = QCheckBox('Mostrar solo modelos .mdl')
        self.only_models.setChecked(True)
        self.only_models.toggled.connect(self.rebuild_tree)
        layout.addWidget(self.only_models)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemDoubleClicked.connect(self.preview_resource)
        layout.addWidget(self.tree)
        for label, callback in [('Usar como arma', lambda: self.use_selected('weapon')),
                                ('Usar como charm', lambda: self.use_selected('charm'))]:
            b = QPushButton(label)
            b.clicked.connect(callback)
            layout.addWidget(b)
        note = QLabel('Abre el VPK del juego o un reemplazo de Workshop. Los archivos compartidos se buscan en las fuentes cargadas.')
        note.setWordWrap(True)
        note.setObjectName('subtle')
        layout.addWidget(note)
        splitter.addWidget(left)
        center = QWidget()
        layout = QVBoxLayout(center)
        self.model_label = QLabel('Arma: sin seleccionar    |    Charm: sin seleccionar')
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        requested = self.settings.value('renderer', 'OpenGL')
        self.viewport = GLViewport() if requested == 'OpenGL' and QApplication.platformName() not in ('offscreen', 'minimal') else Viewport()
        self.connect_viewport(self.viewport)
        layout.addWidget(self.viewport, 1)
        self.viewport_layout = layout
        render_row = QHBoxLayout()
        render_row.addWidget(QLabel('Renderizado'))
        self.renderer = QComboBox()
        self.renderer.addItems(['OpenGL', 'Software', 'Vulkan (no disponible)'])
        self.renderer.model().item(2).setEnabled(False)
        self.renderer.setItemData(2, 'Esta versión de PyQt6 no expone QVulkanWindow ni QRhi.', Qt.ItemDataRole.ToolTipRole)
        self.renderer.setCurrentIndex(0 if isinstance(self.viewport, GLViewport) else 1)
        self.renderer.currentIndexChanged.connect(self.change_renderer)
        render_row.addWidget(self.renderer)
        self.renderer_label = QLabel('OpenGL · iniciando…' if isinstance(self.viewport, GLViewport) else 'Software · compatibilidad')
        self.renderer_label.setObjectName('subtle')
        self.renderer_label.setWordWrap(True)
        render_row.addWidget(self.renderer_label, 1)
        layout.addLayout(render_row)
        QTimer.singleShot(1500, self.check_renderer)
        row = QHBoxLayout()
        self.camera_mode = QComboBox()
        self.camera_mode.addItems(['Órbita', 'Primera persona'])
        self.camera_mode.currentIndexChanged.connect(self.set_camera)
        row.addWidget(self.camera_mode)
        fit = QPushButton('Encuadrar')
        fit.clicked.connect(lambda: self.viewport.fit())
        row.addWidget(fit)
        self.skeleton = QCheckBox('Huesos')
        self.skeleton.setChecked(True)
        self.skeleton.toggled.connect(self.show_bones)
        row.addWidget(self.skeleton)
        wire = QCheckBox('Alambre')
        wire.toggled.connect(self.show_wire)
        row.addWidget(wire)
        row.addStretch()
        row.addWidget(QLabel('FOV'))
        self.fov = QSpinBox()
        self.fov.setRange(15, 120)
        self.fov.setValue(54)
        self.fov.valueChanged.connect(self.set_fov)
        row.addWidget(self.fov)
        layout.addLayout(row)
        anim = QGroupBox('Animaciones del arma')
        box = QVBoxLayout(anim)
        row = QHBoxLayout()
        self.sequence = QComboBox()
        self.sequence.currentIndexChanged.connect(self.change_sequence)
        row.addWidget(self.sequence, 1)
        self.play = QPushButton('▶ Reproducir')
        self.play.setCheckable(True)
        self.play.toggled.connect(lambda active: self.play.setText('Ⅱ Pausar' if active else '▶ Reproducir'))
        row.addWidget(self.play)
        self.speed = QComboBox()
        self.speed.addItems(['0.25×', '0.5×', '1×', '2×'])
        self.speed.setCurrentIndex(2)
        row.addWidget(self.speed)
        box.addLayout(row)
        row = QHBoxLayout()
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.valueChanged.connect(self.scrub)
        row.addWidget(self.timeline, 1)
        self.frame_label = QLabel('0 / 0')
        row.addWidget(self.frame_label)
        box.addLayout(row)
        self.animation_note = QLabel('Las animaciones se leen del MDL / ANI. Vista sólida sin texturas.')
        self.animation_note.setObjectName('subtle')
        self.animation_note.setWordWrap(True)
        box.addWidget(self.animation_note)
        layout.addWidget(anim)
        splitter.addWidget(center)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(310)
        right = QWidget()
        right_scroll.setWidget(right)
        layout = QVBoxLayout(right)
        group = QGroupBox('1  ·  Hueso del arma')
        box = QVBoxLayout(group)
        self.bone_filter = QLineEdit()
        self.bone_filter.setPlaceholderText('Buscar hueso…')
        self.bone_filter.textChanged.connect(self.filter_bones)
        box.addWidget(self.bone_filter)
        self.bones = QTreeWidget()
        self.bones.setHeaderLabels(['Hueso', 'ID'])
        self.bones.setMaximumHeight(220)
        self.bones.setColumnWidth(0, 230)
        self.bones.currentItemChanged.connect(self.bone_changed)
        box.addWidget(self.bones)
        self.bone_note = QLabel('Selecciona un hueso real; el ID 0 es válido.')
        self.bone_note.setWordWrap(True)
        box.addWidget(self.bone_note)
        layout.addWidget(group)
        group = QGroupBox('2  ·  Posición relativa al hueso')
        box = QFormLayout(group)
        self.charm_pose_combo = QComboBox()
        self.charm_pose_combo.addItem('Del juego (secuencia inicial)', 'sequence0')
        self.charm_pose_combo.addItem('De compilación (bind)', 'bind')
        self.charm_pose_combo.setToolTip('El prop de consola reproduce su animación. El VPK integrado conserva exactamente la pose elegida aquí.')
        self.charm_pose_combo.currentIndexChanged.connect(self.change_charm_pose)
        box.addRow('Pose charm',self.charm_pose_combo)
        self.offset_controls, self.angle_controls = [], []
        for label in ('X · adelante', 'Y · izquierda', 'Z · arriba'):
            control = self.spin(-10000, 10000, 0, .05, 10)
            self.offset_controls.append(control)
            box.addRow(label, control)
        for label in ('Pitch', 'Yaw', 'Roll'):
            control = self.spin(-3600, 3600, 0, 1, 180)
            self.angle_controls.append(control)
            box.addRow(label, control)
        self.scale_control = self.spin(.001, 10, 1, .01, 10)
        box.addRow('Escala', self.scale_control)
        hint = QLabel('Shift + arrastrar: ajuste fino ×10. ↺ reinicia solo ese valor.')
        hint.setWordWrap(True)
        hint.setObjectName('subtle')
        box.addRow(hint)
        reset = QPushButton('Restablecer transformación')
        reset.clicked.connect(self.reset_transform)
        box.addRow(reset)
        layout.addWidget(group)
        group = QGroupBox('3  ·  Perfil y salida')
        box = QFormLayout(group)
        self.ident = QLineEdit()
        self.ident.setPlaceholderText('ID del charm')
        self.ident.textChanged.connect(self.refresh_commands)
        self.variant = QLineEdit('default')
        self.variant.textChanged.connect(self.refresh_commands)
        box.addRow('ID charm', self.ident)
        box.addRow('Variante', self.variant)
        for label, callback in [('Guardar perfil JSON', self.save_project), ('Copiar comandos de consola', self.copy_commands),
                                ('Exportar addon natural (sin DLL, experimental)', self.export_natural),
                                ('Exportar charm + CFG (viewmodel nativo)', self.export_native),
                                ('Exportar CFG (prueba servidor)', self.export_cfg),
                                ('Empaquetar charm independiente', self.export_charm)]:
            button = QPushButton(label)
            if 'arma +' in label:
                button.setObjectName('primary')
            button.clicked.connect(callback)
            box.addRow(button)
        layout.addWidget(group)
        note = QLabel('Addon natural experimental: compone brazos + charm sin copiar el arma. Sin DLL ni -insecure; colocación fija al exportar. Requiere Authoring Tools. Las restricciones de addons del servidor se mantienen; conexión remota pendiente de validar.\n\nViewmodel nativo con CFG: requiere extensión local y -insecure. CFG de servidor: usa renderizado del mundo y puede diferir del viewmodel.')
        note.setObjectName('subtle')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.tabs = QTabWidget()
        self.commands = QPlainTextEdit()
        self.commands.setReadOnly(True)
        self.commands.setMaximumHeight(150)
        self.commands.setPlaceholderText('Selecciona arma, charm y hueso para generar comandos.')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(250)
        self.log.setMaximumHeight(150)
        self.tabs.addTab(self.commands, 'Consola')
        self.tabs.addTab(self.log, 'Registro')
        layout.addWidget(self.tabs)
        layout.addStretch()
        splitter.addWidget(right_scroll)
        splitter.setSizes([270, 800, 360])
        self.statusBar().showMessage('Listo · No se modifica ningún archivo del juego al editar')

    def spin(self, low, high, value, step, span):
        control = TransformControl(low, high, value, step, span)
        control.valueChanged.connect(self.transform_changed)
        return control

    def connect_viewport(self, viewport):
        viewport.bonePicked.connect(self.select_bone)
        viewport.surfacePicked.connect(self.place_surface)
        viewport.rendererInfo.connect(self.renderer_ready)
        viewport.renderFailed.connect(lambda reason: QTimer.singleShot(0, lambda: self.renderer_failed(reason)))

    def renderer_ready(self, text):
        self.renderer_label.setText(text)
        self.info(text)

    def check_renderer(self):
        if isinstance(self.viewport, GLViewport) and not self.viewport.isValid():
            self.renderer_failed('No se pudo crear un contexto OpenGL con este controlador')

    def renderer_failed(self, reason):
        if isinstance(self.viewport, GLViewport):
            self.renderer.setCurrentIndex(1)
            self.info('Se activó Software: ' + reason)

    def change_renderer(self, index):
        if index not in (0, 1):
            return
        old = self.viewport
        if (index == 0) == isinstance(old, GLViewport):
            return
        new = GLViewport() if index == 0 else Viewport()
        self.connect_viewport(new)
        new.charm_pose = old.charm_pose
        new.set_models(self.weapon, self.charm)
        for name in ('bone', 'offset', 'angles', 'scale', 'center', 'distance', 'yaw', 'pitch', 'fov', 'first_person', 'skeleton', 'wireframe'):
            setattr(new, name, getattr(old, name))
        if old.pose is not None:
            new.set_pose(old.pose)
        else:
            new.update_charm()
        self.viewport = new
        self.renderer_label.setText('OpenGL · iniciando…' if index == 0 else 'Software · compatibilidad')
        self.viewport_layout.replaceWidget(old, new)
        old.hide()
        old.deleteLater()
        new.show()
        self.settings.setValue('renderer', 'OpenGL' if index == 0 else 'Software')
        if hasattr(self, 'timer'):
            self.timer.setInterval(16 if index == 0 else 33)
        if index == 0:
            QTimer.singleShot(1500, self.check_renderer)

    def info(self, message):
        self.log.appendPlainText(message)
        self.statusBar().showMessage(message, 12000)

    def error(self, error):
        self.info(str(error))
        QMessageBox.warning(self, 'Charm Studio', str(error))

    def submit(self, label, work, callback):
        self.loading += 1
        self.info(label + '…')
        future = self.pool.submit(work)
        future.add_done_callback(lambda result: self.jobFinished.emit((result, callback)))

    def finish_job(self, result):
        self.loading -= 1
        future, callback = result
        try:
            callback(future.result())
        except Exception as error:
            self.error(error)

    def load_defaults(self):
        if not (self.root / 'charms').is_dir():
            self.info('Abre una carpeta o VPK de charms. No se encontró la carpeta charms junto al programa.')
            return
        for directory in sorted((self.root / 'charms').iterdir()):
            if directory.is_dir():
                try:
                    self.resources.mount(directory)
                except Exception as error:
                    self.info(str(error))
        self.rebuild_tree()
        self.info('Charms locales cargados. Selecciona un modelo y pulsa Usar como charm.')

    def open_vpk(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Abrir VPK (elige _dir.vpk en archivos divididos)', self.settings.value('last_folder', str(self.root)), 'Valve VPK (*.vpk)')
        if paths:
            self.settings.setValue('last_folder', str(Path(paths[0]).parent))
            def work():
                return [self.resources.mount(path) for path in paths]
            self.submit('Leyendo VPK', work, self.mounted)

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Carpeta addon (contiene models y materials)', str(self.root))
        if path:
            self.submit('Explorando carpeta', lambda: [self.resources.mount(path)], self.mounted)

    def open_game(self):
        path = QFileDialog.getExistingDirectory(self, 'Instalación de Left 4 Dead 2 o carpeta left4dead2', self.settings.value('game_folder', 'D:/SteamLibrary/steamapps/common'))
        if path:
            self.settings.setValue('game_folder', path)
            self.submit('Cargando paquetes del juego', lambda: self.resources.mount_game(path), self.mounted)

    def mounted(self, mounts):
        self.rebuild_tree()
        self.info('Fuentes cargadas: ' + ', '.join(m.name for m in mounts))

    def rebuild_tree(self):
        self.tree.clear()
        for mount in self.resources.mounts:
            top = QTreeWidgetItem([mount.name])
            top.setToolTip(0, str(mount.path))
            self.tree.addTopLevelItem(top)
            nodes = {'': top}
            for path in sorted(mount.files):
                if self.only_models.isChecked() and not path.endswith('.mdl'):
                    continue
                bits = path.split('/')
                for n in range(1, len(bits)+1):
                    subpath = '/'.join(bits[:n])
                    if subpath not in nodes:
                        item = QTreeWidgetItem(nodes['/'.join(bits[:n-1])], [bits[n-1]])
                        nodes[subpath] = item
                        if n == len(bits):
                            item.setData(0, Qt.ItemDataRole.UserRole, (mount, path))
                            item.setToolTip(0, path)
            top.setExpanded(True)
            # Expanded model paths are useful for the small bundled packs.
            if len(mount.files) < 100:
                for item in nodes.values():
                    item.setExpanded(True)
        self.filter_tree()

    def filter_tree(self):
        text = self.search.text().lower()
        def visit(item):
            data = item.data(0, Qt.ItemDataRole.UserRole)
            match = bool(data and text in data[1])
            children = [visit(item.child(i)) for i in range(item.childCount())]
            visible = match or any(children) or (not text and not data)
            item.setHidden(not visible)
            if text and any(children):
                item.setExpanded(True)
            return visible
        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))

    def selected(self):
        item = self.tree.currentItem()
        data = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if not data:
            raise ValueError('Selecciona un archivo .mdl en el explorador')
        return data

    def preview_resource(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        mount, path = data
        if path.endswith('.mdl'):
            self.use_selected('weapon' if '/v_' in path or '/v_models/' in path else 'charm')
        elif path.endswith(('.nut', '.txt', '.vmt', '.cfg', '.json')):
            try:
                text = mount.read(path)[:100000].decode('utf-8-sig', 'replace')
                dialog = QMessageBox(self)
                dialog.setWindowTitle(path)
                dialog.setText(path)
                dialog.setDetailedText(text)
                dialog.exec()
            except Exception as error:
                self.error(error)

    def use_selected(self, role):
        try:
            mount, path = self.selected()
            if not path.endswith('.mdl'):
                raise ValueError('Selecciona un modelo .mdl')
            def work():
                chosen = self.resources.weapon_source(mount, path) if role == 'weapon' else mount
                return chosen, Model(self.resources.reader(chosen), path)
            def done(result):
                chosen, model = result
                if chosen is not mount:
                    self.rebuild_tree()
                    self.info('Se usa la versión vigente del juego: '+str(chosen.path))
                self.loaded(role, chosen, model)
            self.submit('Cargando ' + path, work, done)
        except Exception as error:
            self.error(error)

    def loaded(self, role, mount, model):
        self.play.setChecked(False)
        if role == 'weapon':
            self.weapon, self.weapon_mount = model, mount
            self.viewport.bone = 0
            self.populate_bones()
            self.sequence.blockSignals(True)
            self.sequence.clear()
            self.sequence.addItem('Pose de referencia (bind pose)', -1)
            for i, seq in enumerate(model.sequences):
                self.sequence.addItem(f'{seq.name} · {seq.frames} frames', i)
            self.sequence.blockSignals(False)
            self.variant.setText(digest(model.data)[:12])
            self.frame = 0
            self.timeline.setRange(0, 0)
        else:
            self.charm, self.charm_mount = model, mount
            self.ident.setText(charm_id(model.path))
        self.viewport.set_models(self.weapon, self.charm)
        self.charm_pose_combo.blockSignals(True)
        self.charm_pose_combo.setCurrentIndex(self.charm_pose_combo.findData(self.viewport.charm_pose))
        self.charm_pose_combo.blockSignals(False)
        if self.viewport.charm_pose_warning:
            self.info(self.viewport.charm_pose_warning+' · Se muestra la pose de compilación.')
        self.apply_frame()
        self.transform_changed()
        self.model_label.setText(f'Arma: {self.weapon.path if self.weapon else "sin seleccionar"}\nCharm: {self.charm.path if self.charm else "sin seleccionar"}\nFuente del arma: {self.weapon_mount.path if self.weapon_mount else "sin seleccionar"}')
        self.info(f'{role}: {len(model.positions)} vértices · {len(model.triangles)} triángulos · {len(model.bones)} huesos')
        for warning in model.warnings:
            self.info(warning)

    def populate_bones(self):
        self.bones.blockSignals(True)
        self.bones.clear()
        items = []
        for i, bone in enumerate(self.weapon.bones):
            item = QTreeWidgetItem([bone.name, str(i)])
            item.setData(0, Qt.ItemDataRole.UserRole, i)
            item.setToolTip(0, f'Padre: {bone.parent} · ID: {i}')
            items.append(item)
        for i, bone in enumerate(self.weapon.bones):
            if 0 <= bone.parent < len(items):
                items[bone.parent].addChild(items[i])
            else:
                self.bones.addTopLevelItem(items[i])
        self.bones.expandAll()
        self.bones.blockSignals(False)
        preferred = next((i for i,b in enumerate(self.weapon.bones) if b.name.lower() in ('valvebiped.weapon_bone', 'weapon_bone', 'v_weapon', 'weapon')), 0)
        if items:
            self.bones.setCurrentItem(items[preferred])

    def all_bone_items(self):
        def walk(item):
            yield item
            for i in range(item.childCount()):
                yield from walk(item.child(i))
        for i in range(self.bones.topLevelItemCount()):
            yield from walk(self.bones.topLevelItem(i))

    def filter_bones(self):
        query = self.bone_filter.text().lower()
        for item in self.all_bone_items():
            item.setHidden(bool(query) and query not in item.text(0).lower())
        if query:
            for item in list(self.all_bone_items()):
                if not item.isHidden():
                    parent = item.parent()
                    while parent:
                        parent.setHidden(False)
                        parent.setExpanded(True)
                        parent = parent.parent()

    def select_bone(self, index):
        for item in self.all_bone_items():
            if item.data(0, Qt.ItemDataRole.UserRole) == index:
                self.bones.setCurrentItem(item)
                self.bones.scrollToItem(item)
                break

    def bone_changed(self, item, previous):
        if item is None or self.weapon is None:
            return
        self.viewport.bone = int(item.data(0, Qt.ItemDataRole.UserRole))
        name = self.weapon.bones[self.viewport.bone].name
        self.bone_note.setText('Hueso activo: ' + name)
        self.transform_changed()

    def place_surface(self, offset):
        for control, value in zip(self.offset_controls, offset):
            control.blockSignals(True)
            control.setValue(float(value))
            control.blockSignals(False)
        self.transform_changed()

    def transform_changed(self):
        if not hasattr(self, 'commands'):
            return
        self.viewport.offset = [c.value() for c in self.offset_controls]
        self.viewport.angles = [c.value() for c in self.angle_controls]
        self.viewport.scale = self.scale_control.value()
        self.viewport.update_charm()
        self.refresh_commands()

    def reset_transform(self):
        for control in self.offset_controls + self.angle_controls:
            control.blockSignals(True)
            control.setValue(0)
            control.blockSignals(False)
        self.scale_control.blockSignals(True)
        self.scale_control.setValue(1)
        self.scale_control.blockSignals(False)
        self.transform_changed()

    def change_charm_pose(self,index):
        previous = self.viewport.charm_pose
        try:
            self.viewport.set_charm_pose(self.charm_pose_combo.currentData())
            self.viewport.charm_pose_warning = ''
            self.refresh_commands()
            self.info('Pose del charm: '+self.charm_pose_combo.currentText()+'. El VPK integrado utiliza esta misma pose.')
        except ValueError as error:
            self.charm_pose_combo.blockSignals(True)
            self.charm_pose_combo.setCurrentIndex(self.charm_pose_combo.findData(previous))
            self.charm_pose_combo.blockSignals(False)
            self.error(error)

    def set_camera(self, index):
        self.viewport.first_person = index == 1
        self.viewport.update()

    def set_fov(self, value):
        self.viewport.fov = value
        self.viewport.update()

    def show_bones(self, value):
        self.viewport.skeleton = value
        self.viewport.update()

    def show_wire(self, value):
        self.viewport.wireframe = value
        self.viewport.update()

    def change_sequence(self, index):
        self.frame = 0
        seq_index = self.sequence.currentData()
        frames = self.weapon.sequences[seq_index].frames if self.weapon and seq_index is not None and seq_index >= 0 else 1
        self.timeline.setRange(0, max(0, (frames-1)*100))
        self.timeline.setValue(0)
        self.apply_frame()

    def scrub(self, value):
        self.frame = value / 100
        self.apply_frame()

    def tick(self):
        now = time.monotonic()
        delta = min(.25, now-self.last_tick)
        self.last_tick = now
        seq_index = self.sequence.currentData()
        if self.play.isChecked() and self.weapon and seq_index is not None and seq_index >= 0:
            seq = self.weapon.sequences[seq_index]
            self.frame = (self.frame + delta*seq.fps*[.25,.5,1,2][self.speed.currentIndex()]) % max(1, seq.frames-1)
            self.timeline.blockSignals(True)
            self.timeline.setValue(int(self.frame*100))
            self.timeline.blockSignals(False)
            self.apply_frame()

    def apply_frame(self):
        if not self.weapon:
            return
        index = self.sequence.currentData()
        if index is None or index < 0:
            pose = self.weapon.bind_pose()
            self.frame_label.setText('Pose base')
            self.animation_note.setText('Pose de referencia. Vista sólida sin texturas; el motor aplica los materiales al jugar.')
        else:
            pose = self.weapon.pose(index, self.frame)
            seq = self.weapon.sequences[index]
            self.frame_label.setText(f'{self.frame:.1f} / {seq.frames-1}')
            self.animation_note.setText(seq.warning or 'Animación leída del modelo. IK, jiggle, capas y blends del motor no se simulan.')
            if not seq.supported:
                self.play.setChecked(False)
        self.viewport.set_pose(pose)

    def placement(self):
        if not self.weapon or not self.charm or not self.weapon.bones:
            raise ValueError('Carga un arma y un charm, y selecciona un hueso')
        return Placement(self.weapon.path, digest(self.weapon.data), self.charm.path, digest(self.charm.data),
                         self.ident.text().strip(), self.weapon.bones[self.viewport.bone].name,
                         [c.value() for c in self.offset_controls], [c.value() for c in self.angle_controls],
                         self.scale_control.value(), self.variant.text().strip(),charm_pose=self.viewport.charm_pose).validate()

    def refresh_commands(self):
        if not hasattr(self, 'commands'):
            return
        try:
            self.commands.setPlainText(self.console_commands())
        except Exception as error:
            self.commands.clear()
            self.commands.setPlaceholderText(str(error))

    def console_commands(self):
        profile=self.placement()
        if profile.charm_pose=='bind' and self.charm.sequences:
            native=reference_vertices(self.charm,'sequence0')
            bound=reference_vertices(self.charm,'bind')
            if len(native) and np.max(np.abs(native-bound))>.001:
                raise ValueError('El CFG usa la pose inicial del juego, distinta de la pose de compilación seleccionada. Elige «Del juego» o exporta el VPK integrado para conservar esta colocación.')
        return profile.commands()

    def copy_commands(self):
        try:
            QApplication.clipboard().setText(self.console_commands())
            self.info('Comandos copiados. Pégalos en un mapa local con la base Studio 0.4.1 y el charm habilitados.')
        except Exception as error:
            self.error(error)

    def save_project(self):
        try:
            profile = self.placement()
            path, _ = QFileDialog.getSaveFileName(self, 'Guardar perfil', str(self.root / 'perfil_charm.json'), 'Perfil JSON (*.json)')
            if path:
                profile.save(path)
                self.info('Perfil guardado: ' + path)
        except Exception as error:
            self.error(error)

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Abrir perfil', str(self.root), 'Perfil JSON (*.json)')
        if not path:
            return
        try:
            profile = Placement.load(path)
            def find(model_path, sha):
                for mount in reversed(self.resources.mounts):
                    if model_path.lower() in mount.files and digest(mount.read(model_path)) == sha:
                        return mount
                raise ValueError(f'Abre primero el VPK o carpeta exactos de {model_path}. La huella del modelo debe coincidir.')
            wm, cm = find(profile.weapon, profile.weapon_sha256), find(profile.charm, profile.charm_sha256)
            def work():
                return Model(self.resources.reader(wm), profile.weapon), Model(self.resources.reader(cm), profile.charm)
            def loaded(models):
                weapon, charm = models
                if profile.bone not in [b.name for b in weapon.bones]:
                    raise ValueError('El hueso del perfil no existe en el modelo')
                self.loaded('weapon', wm, weapon)
                self.loaded('charm', cm, charm)
                self.charm_pose_combo.setCurrentIndex(self.charm_pose_combo.findData(profile.charm_pose))
                self.select_bone([b.name for b in weapon.bones].index(profile.bone))
                for control, value in zip(self.offset_controls+self.angle_controls+[self.scale_control], profile.offset+profile.angles+[profile.scale]):
                    control.setValue(value)
                self.ident.setText(profile.charm_id)
                self.variant.setText(profile.variant)
                self.info('Perfil restaurado y huellas verificadas')
            self.submit('Restaurando perfil', work, loaded)
        except Exception as error:
            self.error(error)

    def export_cfg(self):
        try:
            commands = self.console_commands()
            from .native_profile import profile_filename
            suggested = profile_filename(self.placement())+'_servidor.cfg'
            path, _ = QFileDialog.getSaveFileName(self, 'Exportar configuración', str(self.root / suggested), 'Source config (*.cfg)')
            if path:
                Path(path).write_text('// Requires chains_system_base_studio.vpk 0.4.1 and the charm addon enabled.\n// Local listen-server bone preview. Restart the map after updating addons.\n' + commands + '\n// After spawn: script ChainsSystem.SaveProfile()\n', encoding='ascii')
                self.info('CFG exportado. Cópialo a left4dead2/cfg y ejecuta exec ' + Path(path).stem)
        except Exception as error:
            self.error(error)

    def export_natural(self):
        try:
            from .natural_profile import export_natural_bundle
            profile = self.placement()
            current = self.resources.weapon_source(self.weapon_mount, profile.weapon)
            if digest(current.read(profile.weapon)) != profile.weapon_sha256:
                raise ValueError('Vuelve a cargar el modelo actual del arma antes de exportar.')
            compiler = Path(self.settings.value('native_compiler', r'C:\Program Files (x86)\Steam\steamapps\common\Left 4 Dead 2\bin\studiomdl.exe'))
            if not compiler.is_file():
                selected, _ = QFileDialog.getOpenFileName(self, 'Selecciona studiomdl.exe de L4D2 Authoring Tools', '', 'Source compiler (studiomdl.exe)')
                if not selected:
                    return
                compiler = Path(selected)
            self.settings.setValue('native_compiler', str(compiler))
            folder = QFileDialog.getExistingDirectory(self, 'Exportar VPK natural y perfil editable', str(self.root))
            if not folder:
                return
            weapon, charm = self.weapon, self.charm
            read = self.resources.reader(self.charm_mount)
            resources, mount = self.resources, self.weapon_mount
            def work():
                return export_natural_bundle(profile, weapon, charm, read, resources, mount,
                                              folder, compiler=compiler, workspace=self.root)
            def done(path):
                self.info('Addon natural exportado: '+str(path))
                QMessageBox.information(self, 'Addon natural experimental',
                    'Se exportaron el VPK, el perfil JSON y LEEME_natural.txt.\n\n'
                    'Copia el VPK a addons y reinicia desde Steam. Se activa al equipar el arma, sin CFG ni DLL. '
                    'Incluye brazos + charm para los supervivientes del script; conserva el arma de Workshop.\n\n'
                    'La colocación queda compilada y cambiarla requiere exportar otra vez. Otros reemplazos de brazos/scripts '
                    'pueden entrar en conflicto. Respeta las restricciones de addons del servidor. '
                    'Probado localmente; conexión remota todavía pendiente de validar.')
            self.submit('Compilando brazos y charm por superviviente', work, done)
        except Exception as error:
            self.error(error)

    def export_native(self):
        try:
            from .native_profile import export_native_bundle, profile_filename
            profile = self.placement()
            ident = profile_filename(profile)
            current = self.resources.weapon_source(self.weapon_mount, profile.weapon)
            if digest(current.read(profile.weapon)) != profile.weapon_sha256:
                raise ValueError('El arma cargada es una versión antigua. Selecciona su modelo y pulsa «Usar como arma» para cargar '+str(current.path)+'. Revisa la colocación antes de exportar.')
            compiler = Path(self.settings.value('native_compiler', r'C:\Program Files (x86)\Steam\steamapps\common\Left 4 Dead 2\bin\studiomdl.exe'))
            if not compiler.is_file():
                selected, _ = QFileDialog.getOpenFileName(self, 'Selecciona studiomdl.exe de L4D2 Authoring Tools', '', 'Source compiler (studiomdl.exe)')
                if not selected:
                    return
                compiler = Path(selected)
            self.settings.setValue('native_compiler', str(compiler))
            folder = QFileDialog.getExistingDirectory(self, 'Carpeta para addons, CFG y extensión experimental', str(self.root))
            if not folder:
                return
            weapon, charm = self.weapon, self.charm
            read = self.resources.reader(self.charm_mount)
            plugin = self.root/'native/charms_studio_native.dll'
            def work():
                return export_native_bundle(profile, weapon, charm, read, folder, compiler=compiler, plugin=plugin, workspace=self.root)
            def done(path):
                self.info('CFG del arma exportado: '+str(path/'cfg'/f'{ident}.cfg'))
                QMessageBox.information(self, 'Exportación experimental',
                    'Se exportaron addons, cfg, perfiles y LEEME.txt. Con el arma correspondiente equipada, ejecuta: exec '+ident+'\n\n'
                    'El arma de Workshop sigue proporcionando su modelo y animaciones.\n\n'
                    'Esta extensión funciona en un mapa local de Windows con -insecure y la versión compatible de L4D2. '
                    'Reinicia el juego después de instalar el VPK. Consulta LEEME.txt para cargar y detener el perfil.')
            self.submit('Compilando charm independiente para viewmodel', work, done)
        except Exception as error:
            self.error(error)

    def export_baked(self):
        try:
            profile = self.placement()
            from .bake import bake
            path, _ = QFileDialog.getSaveFileName(self, 'Exportar viewmodel con charm integrado', str(self.root / 'dist/zz_charm_viewmodel.vpk'), 'Valve VPK (*.vpk)')
            if not path:
                return
            weapon_read, charm_read = self.resources.reader(self.weapon_mount), self.resources.reader(self.charm_mount)
            bone_index = self.viewport.bone
            def work():
                files = bake(weapon_read, profile.weapon, charm_read, profile.charm, bone_index, profile.offset, profile.angles, profile.scale,charm_pose=profile.charm_pose)
                files['addoninfo.txt'] = b'"AddonInfo"\n{\n"addonSteamAppID" "550"\n"addonTitle" "Charm Studio - Integrated viewmodel"\n"addonVersion" "1.0"\n"addonAuthor" "Charm Studio"\n}\n'
                # Keep the exact editor record in the package for provenance.
                import json
                from dataclasses import asdict
                files['charmstudio/profile.json'] = json.dumps(asdict(profile), indent=2).encode()
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                write_vpk(path, files)
                return path
            self.submit('Integrando charm en el viewmodel', work, self.baked_done)
        except Exception as error:
            self.error(error)

    def baked_done(self, path):
        self.info('VPK integrado exportado: ' + path)
        QMessageBox.information(self, 'Viewmodel exportado',
            'Copia este VPK a left4dead2/addons y reinicia el mapa.\n\n'
            'Conserva el addon del arma original si usas Workshop: proporciona sus animaciones y materiales compartidos. '
            'El VPK integrado debe tener prioridad sobre su modelo; revisa conflictos en Add-ons.\n\n'
            'Detén la prueba de consola con script ChainsSystem.Stop() para evitar un charm duplicado. '
            'Esta exportación usa LOD0 y deja el charm rígido al hueso, sin física propia. No necesita attachments.')

    def export_charm(self):
        try:
            if not self.charm:
                raise ValueError('Selecciona un charm primero')
            ident = self.ident.text().strip()
            path, _ = QFileDialog.getSaveFileName(self, 'Empaquetar charm independiente', str(self.root / 'dist' / (ident+'.vpk')), 'Valve VPK (*.vpk)')
            if path:
                mount, model_path = self.charm_mount, self.charm.path
                def work():
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    package_charm(mount, model_path, ident, path)
                    return path
                self.submit('Empaquetando charm', work, lambda p: self.info('Charm independiente exportado: ' + p))
        except Exception as error:
            self.error(error)

    def closeEvent(self, event):
        self.timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        event.accept()
