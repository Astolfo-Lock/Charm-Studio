"""Experimental normal VPK export using the native survivor-arms slot."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from .natural_arms import export_natural_arms
from .native_profile import profile_filename
from .project import digest
from .resources import Resources, normalize
from .source import Model
from .vpk import write_vpk


TOKEN = re.compile(rb'//[^\r\n]*|"(?:\\.|[^"\\])*"|[{}]|[^\s{}"]+')


def _tokens(data):
    return [m for m in TOKEN.finditer(data) if not m.group().startswith(b'//')]


def _value(token):
    return token.group().strip(b'"').decode('ascii')


def arms_entries(data):
    """Return original byte spans; keep comments, whitespace and all other keys."""
    tokens = _tokens(data)
    slots = []
    for index, token in enumerate(tokens):
        if _value(token).lower() != 'characterviewmodeladdon':
            continue
        if slots or index+1 >= len(tokens) or tokens[index+1].group() != b'{':
            raise ValueError('Tabla CharacterViewmodelAddon inválida o duplicada')
        cursor = index+2
        seen = set()
        while cursor < len(tokens) and tokens[cursor].group() != b'}':
            if cursor+1 >= len(tokens) or tokens[cursor+1].group() in (b'{', b'}'):
                raise ValueError('Entrada de brazos inválida')
            name, path = _value(tokens[cursor]), normalize(_value(tokens[cursor+1]))
            if name.lower() in seen or not path.startswith('models/') or not path.endswith('.mdl'):
                raise ValueError('Modelo de brazos inválido o superviviente duplicado')
            seen.add(name.lower())
            slots.append((name, path, tokens[cursor+1].span()))
            cursor += 2
        if cursor == len(tokens):
            raise ValueError('Tabla de brazos sin cierre')
    if not slots:
        raise ValueError('El script no contiene CharacterViewmodelAddon')
    return slots


def patch_arms(data, replacements):
    result = data
    entries = arms_entries(data)
    if set(replacements) != {name for name, _, _ in entries}:
        raise ValueError('Faltan variantes de superviviente')
    for name, _, (start, end) in reversed(entries):
        path = normalize(replacements[name])
        if not path.startswith('models/charms_studio/natural/') or not path.endswith('.mdl') or len(path) >= 64:
            raise ValueError('Ruta natural inválida')
        result = result[:start]+b'"'+path.encode('ascii')+b'"'+result[end:]
    return result


def find_weapon_script(resources, preferred, weapon_path):
    """Explicit addon first, then mounted priority; never guess by file stem."""
    mounts = [preferred]+[m for m in reversed(resources.mounts) if m is not preferred]
    visible = {}
    for mount in mounts:
        for path in mount.files:
            if re.fullmatch(r'scripts/weapon_[^/]+\.txt', path):
                visible.setdefault(path, mount)
    matches = []
    for path, mount in visible.items():
        data = mount.read(path)
        tokens = _tokens(data)
        viewmodels = [_value(tokens[i+1]) for i, token in enumerate(tokens[:-1])
                      if _value(token).lower() == 'viewmodel']
        if any(normalize(value) == normalize(weapon_path) for value in viewmodels):
            arms_entries(data)  # fail clearly for unsupported scripts
            matches.append((path, data, mount))
    if not matches:
        if not visible:
            raise ValueError('No hay scripts de armas cargados. Carga la instalación de L4D2 con «Cargar juego» y vuelve a exportar. Modelo: '+weapon_path)
        raise ValueError('Ningún script de arma cargado usa el modelo '+weapon_path+
                         '. Selecciona el viewmodel que reemplaza el arma del juego; los modelos auxiliares y de tercera persona no tienen un script propio.')
    if len(matches) > 1:
        raise ValueError('Varios scripts usan el modelo '+weapon_path+': '+', '.join(path for path, _, _ in matches))
    return matches[0]


def resources_with_game_fallback(resources, game_folder):
    """Resolve official dependencies without mutating UI mounts or shadowing addons.

    Workshop weapon replacements commonly ship only model sidecars/materials.
    Their weapon script and survivor arms still come from the installed game.
    """
    result = Resources()
    result.mount_game(game_folder)
    official_paths = {mount.path for mount in result.mounts}
    # Retain user addon priority above the official update/DLC fallback. Skip
    # already mounted official packs so an old base pack cannot shadow update.
    result.mounts.extend(mount for mount in resources.mounts if mount.path not in official_paths)
    return result


def export_natural_bundle(profile, weapon, charm, charm_read, resources, weapon_mount,
                          output_dir, *, compiler, workspace):
    profile.validate()
    if digest(weapon.data) != profile.weapon_sha256 or digest(charm.data) != profile.charm_sha256:
        raise ValueError('El perfil no coincide con los modelos cargados')
    if profile.bone not in [bone.name for bone in weapon.bones]:
        raise ValueError('El hueso seleccionado no existe en el arma')
    compiler = Path(compiler).resolve()
    if not compiler.is_file():
        raise FileNotFoundError('Selecciona studiomdl.exe de L4D2 Authoring Tools')
    resources = resources_with_game_fallback(resources, compiler.parent.parent)
    script_path, script, script_mount = find_weapon_script(resources, weapon_mount, profile.weapon)
    read = resources.reader(script_mount)
    slots = arms_entries(script)
    files, replacements, variants = {}, {}, []
    # Profile and exact arms input determine a unique short output path.
    for number, (survivor, arms_path, _) in enumerate(slots):
        arms = Model(read, arms_path)
        identity = digest(json.dumps(asdict(profile), sort_keys=True).encode()+arms.data)[:16]
        path = f'models/charms_studio/natural/n_{identity}_{number}.mdl'
        compiled = export_natural_arms(arms, read, charm, charm_read, profile,
                                       path, compiler, workspace)
        for key, data in compiled.items():
            if key in files and files[key] != data:
                raise ValueError('Colisión de recursos en el addon natural: '+key)
            files[key] = data
        replacements[survivor] = path
        variants.append(dict(survivor=survivor, arms=arms_path, model=path,
                             arms_sha256=digest(arms.data)))
    patched = patch_arms(script, replacements)
    files[script_path] = patched
    files['addoninfo.txt'] = (
        '"AddonInfo" { "addonSteamAppID" "550" "addonTitle" "Charm Studio Natural (experimental)" '
        '"addonVersion" "1.5.1" "addonAuthor" "Charm Studio" }\n').encode('ascii')
    ident = profile_filename(profile)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=1, experimental=True, profile=asdict(profile), variants=variants,
                    script=script_path, script_source=str(script_mount.path),
                    script_original_sha256=digest(script), script_patched_sha256=digest(patched),
                    weapon_mesh_included=False, dll_required=False,
                    remote_server_test='inconclusive', files=sorted(files))
    files['charmstudio/natural_manifest.json'] = json.dumps(manifest, indent=2).encode('utf8')
    write_vpk(output/f'{ident}_natural.vpk', files)
    profile.save(output/f'{ident}_natural.json')
    (output/'LEEME_natural.txt').write_text(
        'CHARM STUDIO: ADDON NATURAL EXPERIMENTAL\n\n'
        'Copia el VPK a left4dead2/addons y reinicia el juego. Conserva tu arma de Workshop.\n'
        'Se activa automáticamente para el arma del perfil; no necesita exec, DLL, plugin_load ni -insecure.\n'
        'Inicia el juego normalmente desde Steam. Cierra cualquier prueba anterior que cargara la DLL.\n\n'
        'El VPK contiene brazos + charm para cada superviviente, sin copiar el modelo del arma.\n'
        'Usa el único espacio nativo de brazos del arma. Otros mods de brazos/scripts pueden entrar en conflicto.\n'
        'Usa los brazos definidos en el script montado; los materiales siguen viniendo del juego/addons originales.\n'
        'El perfil queda compilado: ajustar posición, escala o hueso exige volver a exportar y reiniciar.\n'
        'Un perfil por arma. Retira la exportación anterior de esa arma antes de instalar otra.\n'
        'Brazos LOD0, sin variantes de bodygroup/skin. Pose inicial del charm y sus parámetros jiggle conservados.\n\n'
        'Prueba confirmada: partida local normal con sv_cheats 0 y sin plugins.\n'
        'La prueba cliente-servidor separado fue inconclusa; compatibilidad remota aún no certificada.\n'
        'Las restricciones de addons del servidor siguen aplicándose. No garantiza todos los modos/servidores.\n'
        'Para desinstalar, quita solo este VPK y reinicia. El JSON conserva la colocación editable.\n', encoding='utf8')
    return output/f'{ident}_natural.vpk'
