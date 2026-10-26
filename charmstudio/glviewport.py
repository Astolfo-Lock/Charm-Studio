"""Indexed OpenGL rendering through Qt's bindings; no per-triangle Python drawing."""
import time
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMatrix4x4, QVector3D, QPainter, QSurfaceFormat, QColor
from PyQt6.QtOpenGL import QOpenGLFunctions_2_0, QOpenGLShader, QOpenGLShaderProgram, QOpenGLBuffer
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from .viewport import ViewportState

GL_ARRAY_BUFFER = 0x8892
GL_ELEMENT_ARRAY_BUFFER = 0x8893
GL_STATIC_DRAW = 0x88E4
GL_DYNAMIC_DRAW = 0x88E8
GL_FLOAT = 0x1406
GL_UNSIGNED_INT = 0x1405
GL_DEPTH_TEST = 0x0B71

VERTEX_SHADER = '''#version 120
attribute vec3 position;
uniform mat4 mvp;
varying vec3 worldPosition;
void main() {
    worldPosition = position;
    gl_Position = mvp * vec4(position, 1.0);
}
'''
FRAGMENT_SHADER = '''#version 120
uniform vec3 color;
varying vec3 worldPosition;
void main() {
    vec3 face = cross(dFdx(worldPosition), dFdy(worldPosition));
    float intensity = 0.4 + 0.6 * abs(dot(normalize(face), vec3(0.3, -0.4, 0.8660254)));
    gl_FragColor = vec4(color * intensity, 1.0);
}
'''


class GLViewport(ViewportState, QOpenGLWidget):
    bonePicked = pyqtSignal(int)
    surfacePicked = pyqtSignal(object)
    rendererInfo = pyqtSignal(str)
    renderFailed = pyqtSignal(str)
    backend_name = 'OpenGL'

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSwapInterval(1)
        self.setFormat(fmt)
        self.gl = None
        self.program = None
        self.meshes = {}
        self.failure = ''
        self.draw_calls = 0
        self.last_frame_ms = 0.
        self.driver = ''

    def paintEvent(self, event):
        # ViewportState's implementation is reserved for the software backend.
        QOpenGLWidget.paintEvent(self, event)

    def initializeGL(self):
        try:
            self.gl = QOpenGLFunctions_2_0()
            if not self.gl.initializeOpenGLFunctions():
                raise RuntimeError('El controlador no proporciona OpenGL 2.0')
            self.program = QOpenGLShaderProgram(self)
            for kind, source in ((QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER),
                                 (QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SHADER)):
                if not self.program.addShaderFromSourceCode(kind, source):
                    raise RuntimeError(self.program.log())
            self.program.bindAttributeLocation('position', 0)
            if not self.program.link():
                raise RuntimeError(self.program.log())
            self.driver = f'{self.gl.glGetString(0x1F01)} · {self.gl.glGetString(0x1F02)}'
            self.rendererInfo.emit('OpenGL · ' + self.driver)
            self.context().aboutToBeDestroyed.connect(self.cleanup)
        except Exception as error:
            self.fail(error)

    def fail(self, error):
        if not self.failure:
            self.failure = str(error)
            self.renderFailed.emit(self.failure)

    def matrix(self):
        eye, forward, right, up, _ = self.camera()
        projection = QMatrix4x4()
        projection.perspective(self.fov, self.width()/max(1,self.height()), .05, max(10000., self.distance*50))
        view = QMatrix4x4()
        view.lookAt(QVector3D(*map(float, eye)), QVector3D(*map(float, eye+forward)), QVector3D(*map(float, up)))
        return projection * view

    def upload_mesh(self, role, model, vertices):
        gl = self.gl
        mesh = self.meshes.get(role)
        if mesh is None:
            vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            ibo = QOpenGLBuffer(QOpenGLBuffer.Type.IndexBuffer)
            if not vbo.create() or not ibo.create():
                raise RuntimeError('No se pudieron crear los buffers de OpenGL')
            vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw)
            ibo.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
            mesh = {'vbo': vbo, 'ibo': ibo, 'model': None, 'vertices': None}
            self.meshes[role] = mesh
        if mesh['model'] is not model:
            indices = np.ascontiguousarray(model.triangles, dtype=np.uint32)
            mesh['ibo'].bind()
            mesh['ibo'].allocate(indices.tobytes(), indices.nbytes)
            error = gl.glGetError()
            if error:
                raise RuntimeError(f'OpenGL index upload error: 0x{error:04x}, {indices.nbytes} bytes')
            mesh['count'] = indices.size
            mesh['model'] = model
        if mesh['vertices'] is not vertices:
            array = np.ascontiguousarray(vertices, dtype=np.float32)
            mesh['vbo'].bind()
            mesh['vbo'].allocate(array.tobytes(), array.nbytes)
            error = gl.glGetError()
            if error:
                raise RuntimeError(f'OpenGL vertex upload error: 0x{error:04x}, {array.nbytes} bytes')
            mesh['vertices'] = vertices
        return mesh

    def paintGL(self):
        if self.failure or self.program is None:
            return
        started = time.perf_counter()
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor('#101722'))
            camera = self.camera()
            self.draw_grid(painter, camera)
            painter.beginNativePainting()
            gl = self.gl
            ratio = self.devicePixelRatioF()
            gl.glViewport(0, 0, round(self.width()*ratio), round(self.height()*ratio))
            gl.glDepthMask(True)
            gl.glColorMask(True, True, True, True)
            gl.glDisable(0x0C11)  # scissor left by the Qt paint engine
            gl.glClear(0x00000100)  # depth only: keep QPainter background/grid
            gl.glEnable(GL_DEPTH_TEST)
            gl.glDepthFunc(0x0203)  # LEQUAL
            gl.glDisable(0x0B44)  # don't cull two-sided viewmodel parts
            gl.glPolygonMode(0x0408, 0x1B01 if self.wireframe else 0x1B02)
            self.program.bind()
            self.program.setUniformValue('mvp', self.matrix())
            self.program.enableAttributeArray(0)
            self.draw_calls = 0
            for role, model, vertices, color in (
                ('weapon', self.weapon, self.weapon_vertices, (112/255,136/255,162/255)),
                ('charm', self.charm, self.charm_vertices, (240/255,170/255,74/255)),
            ):
                if model is None or not len(vertices) or not len(model.triangles):
                    continue
                mesh = self.upload_mesh(role, model, vertices)
                mesh['vbo'].bind()
                mesh['ibo'].bind()
                self.program.setAttributeBuffer(0, GL_FLOAT, 0, 3, 12)
                self.program.setUniformValue('color', *color)
                gl.glDrawElements(0x0004, mesh['count'], GL_UNSIGNED_INT, None)
                error = gl.glGetError()
                if error:
                    raise RuntimeError(f'OpenGL draw error: 0x{error:04x}')
                self.draw_calls += 1
            self.program.disableAttributeArray(0)
            self.program.release()
            gl.glBindBuffer(GL_ARRAY_BUFFER, 0)
            gl.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
            gl.glPolygonMode(0x0408, 0x1B02)
            gl.glDisable(GL_DEPTH_TEST)
            painter.endNativePainting()
            self.draw_overlay(painter, camera)
        except Exception as error:
            if painter.isActive():
                painter.endNativePainting()
            self.fail(error)
        finally:
            painter.end()
            self.last_frame_ms = (time.perf_counter()-started)*1000

    def cleanup(self):
        if self.gl is None or not self.context() or not self.context().isValid():
            return
        self.makeCurrent()
        for mesh in self.meshes.values():
            mesh['vbo'].destroy()
            mesh['ibo'].destroy()
        self.meshes.clear()
        if self.program:
            self.program.removeAllShaders()
        self.program = None
        self.doneCurrent()
