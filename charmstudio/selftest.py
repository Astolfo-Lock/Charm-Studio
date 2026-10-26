"""Deterministic --self-test for the portable executable, without opening a UI."""
import json
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase
from .window import MainWindow
from .source import Model
from .bake import bake
from .vpk import VPK, write_vpk
from .glviewport import GLViewport


def run(app, root, gpu=False):
    # The offscreen Qt platform doesn't enumerate the Windows system fonts.
    if not QFontDatabase.families():
        for name in ('segoeui.ttf', 'segoeuib.ttf'):
            QFontDatabase.addApplicationFont('C:/Windows/Fonts/' + name)
    output = Path(root) / 'diagnostico'
    output.mkdir(exist_ok=True)
    window = MainWindow(root)
    result_path = output/('resultado_gpu.json' if gpu else 'resultado.json')
    try:
        if gpu:
            window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            window.renderer.setCurrentIndex(0)
        window.show()
        app.processEvents()
        assert len(window.resources.mounts) >= 2
        wm, cm = window.resources.mounts[:2]
        wp = next(p for p in wm.files if p.endswith('.mdl'))
        cp = next(p for p in cm.files if p.endswith('.mdl'))
        wr, cr = window.resources.reader(wm), window.resources.reader(cm)
        weapon, charm = Model(wr, wp), Model(cr, cp)
        window.loaded('weapon', wm, weapon)
        window.loaded('charm', cm, charm)
        window.select_bone(1)
        window.offset_controls[0].setValue(1.25)
        window.scale_control.setValue(.3)
        window.offset_controls[1].setValue(4)
        window.offset_controls[1].reset_button.click()
        assert window.offset_controls[1].value() == 0
        assert window.viewport.bone == 1
        profile = window.placement()
        profile.save(output/'perfil.json')
        assert profile.commands().startswith('script_execute')
        files = bake(wr, wp, cr, cp, 1, profile.offset, profile.angles, profile.scale)
        archive = output/'fixture_integrado_no_instalar.vpk'
        write_vpk(archive, files)
        combined = Model(VPK(archive).read, wp)
        assert len(combined.triangles) == len(weapon.triangles)+len(charm.triangles)
        app.processEvents()
        if gpu:
            assert isinstance(window.viewport, GLViewport), 'No se activó OpenGL'
            window.viewport.grabFramebuffer()
            assert window.viewport.isValid(), 'Contexto OpenGL no disponible'
            assert not window.viewport.failure, window.viewport.failure
            assert window.viewport.draw_calls == 2, 'No se dibujaron ambas mallas'
            pose = window.viewport.pose.copy()
            window.renderer.setCurrentIndex(1)
            assert window.viewport.bone == 1 and window.viewport.offset[0] == 1.25
            window.renderer.setCurrentIndex(0)
            app.processEvents()
            window.viewport.grabFramebuffer()
            assert not window.viewport.failure, window.viewport.failure
            import numpy as np
            np.testing.assert_allclose(window.viewport.pose, pose)
            assert window.viewport.draw_calls == 2
            assert window.viewport.driver in window.renderer_label.text(), 'Falta el controlador en la interfaz'
        window.grab().save(str(output/'interfaz.png'))
        result = {'ok': True, 'vertices': len(combined.positions), 'triangles': len(combined.triangles),
                  'scope': 'offline fixture test; not an L4D2 runtime test'}
        if gpu:
            result['driver'] = window.viewport.driver
            result['draw_calls'] = window.viewport.draw_calls
            result['renderer_switch'] = 'state preserved'
        result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        return 0
    except Exception as error:
        result_path.write_text(json.dumps({'ok': False, 'error': repr(error)}, indent=2), encoding='utf-8')
        return 1
    finally:
        window.close()
        app.processEvents()
