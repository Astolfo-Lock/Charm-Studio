"""Shared scene controls and software fallback for the accelerated viewport."""
import math
import numpy as np
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF, QFont
from PyQt6.QtWidgets import QWidget
from .transforms import transform, points
from .charm_pose import reference_vertices


class ViewportState:
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(460, 360)
        self.weapon = self.charm = None
        self.pose = None
        self.weapon_vertices = np.empty((0, 3))
        self.charm_vertices = np.empty((0, 3))
        self.bone = 0
        self.offset = [0., 0., 0.]
        self.angles = [0., 0., 0.]
        self.scale = 1.
        self.center = np.zeros(3)
        self.distance = 50.
        self.yaw, self.pitch = 135., 20.
        self.fov = 54.
        self.first_person = False
        self.skeleton = True
        self.wireframe = False
        self.last_mouse = None
        self.drag_distance = 0
        self.bone_screen = []
        self.camera_cache = None
        self.charm_bind_vertices = np.empty((0, 3))
        self.charm_pose = 'sequence0'
        self.charm_pose_warning = ''
        self.setMouseTracking(True)

    def set_models(self, weapon, charm):
        changed = weapon is not self.weapon
        charm_changed = charm is not self.charm
        self.weapon, self.charm = weapon, charm
        if charm_changed:
            try:
                self.set_charm_pose(self.charm_pose)
            except ValueError as error:
                self.charm_pose_warning = str(error)
                self.set_charm_pose('bind')
        if weapon:
            self.pose = weapon.bind_pose()
            self.weapon_vertices = weapon.skinned_vertices(self.pose)
        else:
            self.pose = None
            self.weapon_vertices = np.empty((0, 3))
        self.update_charm()
        if changed or not weapon:
            self.fit()

    def set_charm_pose(self, mode):
        vertices = reference_vertices(self.charm, mode) if self.charm else np.empty((0,3))
        self.charm_pose = mode
        self.charm_bind_vertices = vertices
        self.update_charm()

    def set_pose(self, pose):
        self.pose = pose
        self.weapon_vertices = self.weapon.skinned_vertices(pose)
        self.update_charm()

    def update_charm(self):
        if self.charm:
            matrix = transform(self.offset, self.angles, self.scale)
            if self.pose is not None and 0 <= self.bone < len(self.pose):
                matrix = self.pose[self.bone] @ matrix
            self.charm_vertices = points(matrix, self.charm_bind_vertices)
        else:
            self.charm_vertices = np.empty((0, 3))
        self.update()

    def fit(self):
        arrays = [v for v in (self.weapon_vertices, self.charm_vertices) if len(v)]
        verts = np.vstack(arrays) if arrays else np.empty((0, 3))
        if len(verts):
            self.center = (verts.min(axis=0) + verts.max(axis=0)) / 2
            self.distance = max(2., np.linalg.norm(np.ptp(verts, axis=0)) * 1.4)
        self.update()

    def camera(self):
        if self.first_person:
            eye = np.zeros(3)
            forward, right, up = np.array([1., 0, 0]), np.array([0., -1, 0]), np.array([0., 0, 1])
        else:
            yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
            direction = np.array([math.cos(pitch)*math.cos(yaw), math.cos(pitch)*math.sin(yaw), math.sin(pitch)])
            eye = self.center + direction * self.distance
            forward = -direction
            right = np.cross(forward, [0., 0., 1.])
            right /= max(1e-12, np.linalg.norm(right))
            up = np.cross(right, forward)
        focal = self.height() / (2 * math.tan(math.radians(self.fov / 2)))
        return eye, forward, right, up, focal

    def project(self, verts, camera):
        eye, forward, right, up, focal = camera
        relative = verts - eye
        depth = relative @ forward
        safe = np.maximum(depth, .01)
        screen = np.column_stack((self.width()/2 + focal*(relative @ right)/safe,
                                  self.height()/2 - focal*(relative @ up)/safe))
        return screen, depth

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#101722'))
        camera = self.camera()
        self.camera_cache = camera
        self.draw_grid(painter, camera)
        self.draw_software_meshes(painter, camera)
        self.draw_overlay(painter, camera)
        painter.end()

    def draw_grid(self, painter, camera):
        # Ground grid in Source XY, Z up.
        grid = []
        size = max(1., self.distance / 20)
        origin = np.round(self.center / size) * size
        for i in range(-10, 11):
            grid.extend([[origin[0]+i*size, origin[1]-10*size, 0], [origin[0]+i*size, origin[1]+10*size, 0],
                         [origin[0]-10*size, origin[1]+i*size, 0], [origin[0]+10*size, origin[1]+i*size, 0]])
        grid_screen, grid_depth = self.project(np.array(grid), camera)
        painter.setPen(QPen(QColor('#202b3a'), 1))
        for i in range(0, len(grid), 2):
            if grid_depth[i] > .1 and grid_depth[i+1] > .1:
                a, b = grid_screen[i:i+2]
                if np.max(np.abs([a,b])) < 1e5:
                    painter.drawLine(QPointF(*a), QPointF(*b))
    def draw_software_meshes(self, painter, camera):
        batches = []
        for model, verts, color in ((self.weapon, self.weapon_vertices, (112, 136, 162)),
                                     (self.charm, self.charm_vertices, (240, 170, 74))):
            if model is None or not len(verts) or not len(model.triangles):
                continue
            screen, depth = self.project(verts, camera)
            tri = model.triangles
            coords = screen[tri]
            valid = (depth[tri].min(axis=1) > .05) & (coords[:,:,0].max(axis=1) >= 0) & (coords[:,:,0].min(axis=1) <= self.width()) & (coords[:,:,1].max(axis=1) >= 0) & (coords[:,:,1].min(axis=1) <= self.height())
            indices = np.flatnonzero(valid)
            faces = verts[tri[indices]]
            normals = np.cross(faces[:,1]-faces[:,0], faces[:,2]-faces[:,0])
            normals /= np.maximum(1e-12, np.linalg.norm(normals, axis=1))[:,None]
            light = .4 + .6 * np.abs(normals @ np.array([.3, -.4, .866]))
            for n, index in enumerate(indices):
                batches.append((float(depth[tri[index]].mean()), coords[index], tuple(int(min(255, v*light[n])) for v in color)))
        batches.sort(key=lambda x: x[0], reverse=True)
        painter.setPen(Qt.PenStyle.NoPen)
        for _, vertices, color in batches:
            if self.wireframe:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(*color), .6))
            else:
                painter.setBrush(QColor(*color))
            painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x,y in vertices]))
    def draw_overlay(self, painter, camera):
        self.bone_screen = []
        if self.weapon and self.pose is not None:
            bone_points = self.pose[:,:3,3]
            screen, depth = self.project(bone_points, camera)
            self.bone_screen = [(i, point) for i, point in enumerate(screen) if depth[i] > .05]
            if self.skeleton:
                for i, bone in enumerate(self.weapon.bones):
                    if bone.parent >= 0 and depth[i] > .05 and depth[bone.parent] > .05:
                        painter.setPen(QPen(QColor('#53bdba') if i != self.bone else QColor('#f9d16e'), 1.5))
                        painter.drawLine(QPointF(*screen[i]), QPointF(*screen[bone.parent]))
                for i, point in self.bone_screen:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor('#ffe4a0') if i == self.bone else QColor('#67d5d0'))
                    painter.drawEllipse(QPointF(*point), 4 if i == self.bone else 2.5, 4 if i == self.bone else 2.5)
            if 0 <= self.bone < len(self.pose):
                anchor = self.pose[self.bone] @ transform(self.offset, self.angles)
                axes = np.vstack([anchor[:3,3], *(anchor[:3,3] + anchor[:3,i]*self.distance*.06 for i in range(3))])
                projected, depths = self.project(axes, camera)
                if depths.min() > .05:
                    for i, (color, label) in enumerate((('#fa7272','X'),('#7bd591','Y'),('#73a7ff','Z'))):
                        painter.setPen(QPen(QColor(color), 2.5))
                        painter.drawLine(QPointF(*projected[0]), QPointF(*projected[i+1]))
                        painter.drawText(QPointF(*(projected[i+1]+[4,-4])), label)
        painter.setPen(QColor('#8595aa'))
        painter.setFont(QFont('Segoe UI', 9))
        painter.drawText(18, 26, 'PRIMERA PERSONA · FOV ' + str(int(self.fov)) if self.first_person else 'VISTA 3D · ÓRBITA')
        painter.drawText(18, self.height()-18, 'Arrastrar: orbitar   ·   Derecho: desplazar   ·   Rueda: zoom   ·   Shift + clic: colocar sobre el arma')
        if not self.weapon and not self.charm:
            painter.setPen(QColor('#d9e4f3'))
            painter.setFont(QFont('Segoe UI', 16))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Abre un VPK o una carpeta de modelos\npara comenzar')

    def mousePressEvent(self, event):
        self.last_mouse = event.position()
        self.drag_distance = 0

    def mouseMoveEvent(self, event):
        if self.last_mouse is None or event.buttons() == Qt.MouseButton.NoButton:
            return
        delta = event.position() - self.last_mouse
        self.drag_distance += abs(delta.x()) + abs(delta.y())
        self.last_mouse = event.position()
        if event.buttons() & Qt.MouseButton.RightButton:
            camera = self.camera()
            self.center += (-camera[2]*delta.x()+camera[3]*delta.y()) * self.distance / max(100, self.height())
        else:
            self.yaw -= delta.x()*.4
            self.pitch = max(-89, min(89, self.pitch+delta.y()*.4))
        self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_distance < 5 and event.button() == Qt.MouseButton.LeftButton:
            point = np.array([event.position().x(), event.position().y()])
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.pick_surface(point)
            elif self.skeleton and self.bone_screen:
                index, projected = min(self.bone_screen, key=lambda pair: np.linalg.norm(pair[1]-point))
                if np.linalg.norm(projected-point) < 14:
                    self.bonePicked.emit(index)
        self.last_mouse = None

    def wheelEvent(self, event):
        if self.first_person:
            self.fov = max(15, min(120, self.fov-event.angleDelta().y()/120*2))
        else:
            self.distance = max(.1, min(100000, self.distance * math.exp(-event.angleDelta().y()/120*.12)))
        self.update()

    def pick_surface(self, screen):
        if self.weapon is None or self.pose is None or not len(self.weapon.triangles):
            return
        eye, forward, right, up, focal = self.camera()
        ray = forward + right*(screen[0]-self.width()/2)/focal - up*(screen[1]-self.height()/2)/focal
        ray /= np.linalg.norm(ray)
        triangles = self.weapon_vertices[self.weapon.triangles]
        edge1, edge2 = triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]
        h = np.cross(np.broadcast_to(ray, edge2.shape), edge2)
        det = np.einsum('ij,ij->i', edge1, h)
        safe = np.where(np.abs(det)>1e-9, det, np.inf)
        s = eye-triangles[:,0]
        u = np.einsum('ij,ij->i', s, h)/safe
        q = np.cross(s, edge1)
        v = q @ ray / safe
        t = np.einsum('ij,ij->i', edge2, q)/safe
        valid = (np.abs(det)>1e-9)&(u>=0)&(v>=0)&(u+v<=1)&(t>.01)
        if valid.any():
            distance = t[np.where(valid)[0]].min()
            local = np.linalg.inv(self.pose[self.bone]) @ np.append(eye + ray*distance, 1)
            self.surfacePicked.emit(local[:3].tolist())


class Viewport(ViewportState, QWidget):
    bonePicked = pyqtSignal(int)
    surfacePicked = pyqtSignal(object)
    rendererInfo = pyqtSignal(str)
    renderFailed = pyqtSignal(str)
    backend_name = 'Software'
