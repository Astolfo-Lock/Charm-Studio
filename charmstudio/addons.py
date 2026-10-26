"""Self-registering independent charm packages."""
from pathlib import Path
import re
from .project import array
from .vpk import write_vpk
from .materials import repair_material
from .runtime_contract import RUNTIME_PATH


def package_charm(mount, model_path, ident, destination):
    if not re.fullmatch(r'[a-zA-Z0-9_\-]{1,64}', ident):
        raise ValueError('ID del addon: usa de 1 a 64 letras, números, guiones o guiones bajos')
    # Pack all materials/models from this charm source; retains shader patches and
    # dependencies that cannot be discovered reliably from just $basetexture.
    files = {}
    for path in mount.files:
        if path.startswith(('models/', 'materials/')):
            files[path] = repair_material(path, mount.read(path), mount.read)[0]
    if model_path.lower() not in files:
        raise ValueError('El modelo debe pertenecer a la fuente del charm')
    script_name = f'charms_system/packs/{ident}'
    files[f'scripts/vscripts/{script_name}.nut'] = (
        f'IncludeScript("{RUNTIME_PATH}/bootstrap", getroottable());\n'
        f'::ChainsSystem.StudioRegister({array(ident)},{array(model_path)});\n'
    ).encode('utf-8')
    files['scripts/vscripts/director_base_addon.nut'] = f'IncludeScript("{script_name}", getroottable());\n'.encode()
    files['addoninfo.txt'] = (
        '"AddonInfo"\n{\n"addonSteamAppID" "550"\n'
        f'"addonTitle" "Charm - {ident}"\n"addonVersion" "1.0"\n'
        '"addonAuthor" "Charm Studio"\n"addonDescription" "Independent Chains System charm pack."\n}\n'
    ).encode()
    write_vpk(destination, files)
