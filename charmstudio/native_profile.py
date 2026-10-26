"""Experimental local native runtime bundle; weapon assets stay in Workshop."""
import json
from pathlib import Path
import re
import shutil
import struct
import tempfile
from .native_adapter import export_native_adapter
from .vpk import VPK, write_vpk


WEAPON_LABELS = {
    'v_rifle_ak47': 'Ak', 'v_rifle': 'M16', 'v_rifle_desert': 'Scar',
    'v_rifle_sg552': 'SG552', 'v_smg': 'Uzi', 'v_smg_silenced': 'Mac10',
    'v_smg_mp5': 'MP5', 'v_pistola': 'Pistola', 'v_pistol': 'Pistol',
    'v_desert_eagle': 'Magnum', 'v_pumpshotgun': 'Pump',
    'v_shotgun_chrome': 'Chrome', 'v_autoshotgun': 'AutoShotgun',
    'v_shotgun_spas': 'Spas', 'v_huntingrifle': 'Hunting',
    'v_sniper_military': 'SniperMilitar', 'v_sniper_scout': 'Scout',
    'v_sniper_awp': 'AWP', 'v_m60': 'M60', 'v_grenade_launcher': 'Granadas',
}


def _slug(value):
    return re.sub('[^a-zA-Z0-9_]', '_', value).strip('_')[:48] or 'modelo'


def charm_label(profile):
    from .project import charm_id
    name = Path(profile.charm).stem if profile.charm_id == charm_id(profile.charm) else profile.charm_id
    return _slug(name.removeprefix('charm_'))


def profile_filename(profile):
    profile.validate()
    stem = Path(profile.weapon).stem.lower()
    weapon = WEAPON_LABELS.get(stem, _slug(stem.removeprefix('v_')))
    return f'charm_{charm_label(profile)}_{weapon}'


def native_commands(profile, weapon, model_path, script_path):
    profile.validate()
    if any(c in profile.weapon for c in '";') or not re.fullmatch(r'charms_studio/native/[a-zA-Z0-9_]+', script_path):
        raise ValueError('Identificador de consola nativo no valido')
    if weapon.path != profile.weapon.lower():
        raise ValueError('El arma cargada no coincide con el perfil')
    from .project import digest
    if digest(weapon.data) != profile.weapon_sha256:
        raise ValueError('La huella del arma no coincide con el perfil')
    names = [bone.name for bone in weapon.bones]
    bone = names.index(profile.bone)
    checksum = struct.unpack_from('<I', weapon.data, 8)[0]
    values = ' '.join(f'{x:.6f}' for x in profile.offset+profile.angles+[profile.scale])
    commands = [
        'sv_cheats 1',
        'plugin_load addons/charms_studio_native',
        f'cmd cs_charm_weapon {checksum} "{profile.weapon}"',
        f'cmd cs_charm_pose {bone} {values}',
        f'script_execute {script_path}',
        'gameui_hide', 'hideconsole',
    ]
    if any(len(line.encode('ascii')) > 240 for line in commands):
        raise ValueError('El comando nativo supera 240 bytes')
    return '\n'.join(commands)+'\n'


def spawn_script(model_path):
    if not re.fullmatch(r'models/charms_studio/native/(?:p_)?[a-f0-9]+\.mdl', model_path):
        raise ValueError('Ruta de charm nativo no valida')
    model = json.dumps(model_path)
    return f'''// Charm-only client viewmodel experiment, local listen server.
local host=GetListenServerHost();
if(host==null)throw "Inicia un mapa local antes de ejecutar el perfil";
if(("CSNative" in getroottable())&&CSNative.entity!=null&&CSNative.entity.IsValid())
    DoEntFire("!self","Kill","",0,null,CSNative.entity);
if(("ChainsSystem" in getroottable())&&("Stop" in ChainsSystem))ChainsSystem.Stop();
PrecacheModel({model});
local original=NetProps.GetPropEntity(host,"m_hViewModel");
if(original==null)throw "No hay viewmodel activo";
local e=SpawnEntityFromTable("predicted_viewmodel",{{solid=0}});
if(e==null)throw "No se pudo crear el viewmodel del charm";
::CSNative<-{{entity=e}};
e.SetModel({model});
NetProps.SetPropEntity(e,"m_hOwner",host);
NetProps.SetPropEntity(e,"m_hWeapon",null);
NetProps.SetPropInt(e,"m_nViewModelIndex",0);
NetProps.SetPropInt(e,"m_fEffects",513);
NetProps.SetPropInt(e,"m_nSequence",0);
DoEntFire("!self","SetParent","!activator",0,original,e);
printl("[Charm Studio native] Perfil cargado; requiere extension cliente local");
'''


def export_native_bundle(profile, weapon, charm, charm_read, output_dir, *, compiler, plugin, workspace):
    plugin = Path(plugin)
    if not plugin.is_file():
        raise FileNotFoundError('No se encontro charms_studio_native.dll junto al programa')
    output = Path(output_dir)
    ident = profile_filename(profile)
    # Validate before starting the compiler or writing output.
    native_commands(profile, weapon, 'models/charms_studio/native/preview.mdl', 'charms_studio/native/preview')
    staging = Path(workspace)/'.build/native_bundle'
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging) as temporary:
        result = export_native_adapter(profile, charm, charm_read, Path(temporary)/'charm.vpk', compiler=compiler, workspace=workspace)
        archive = VPK(result.vpk_path)
        files = {path: archive.read(path) for path in archive.entries}
        script_path = 'charms_studio/native/'+Path(result.model_path).stem
        files['scripts/vscripts/'+script_path+'.nut'] = spawn_script(result.model_path).encode('ascii')
        # Geometry is independent of the weapon and placement. Reuse one pack
        # for identical charm poses; retain older poses referenced by other CFGs.
        pack = f'charm_{charm_label(profile)}_{Path(result.model_path).stem}.vpk'
        (output/'addons').mkdir(parents=True, exist_ok=True)
        write_vpk(output/'addons'/pack, files)
    (output/'cfg').mkdir(exist_ok=True)
    (output/'cfg'/f'{ident}.cfg').write_text('// Experimental Windows local runtime; launch with -insecure.\n'+native_commands(profile, weapon, result.model_path, script_path), encoding='ascii')
    shutil.copy2(plugin, output/'addons/charms_studio_native.dll')
    (output/'perfiles').mkdir(exist_ok=True)
    profile.save(output/'perfiles'/f'{ident}.json')
    configs = sorted(path.stem for path in (output/'cfg').glob('charm_*.cfg'))
    (output/'LEEME.txt').write_text(
        'PRUEBA EXPERIMENTAL WINDOWS, SERVIDOR LOCAL\n\n'
        '1. Copia addons y cfg dentro de left4dead2. Conserva tu arma de Workshop.\n'
        '2. Inicia L4D2 con -insecure. Esta extension impide entrar a servidores seguros.\n'
        '3. Inicia un mapa local y ejecuta el CFG del arma que llevas (activa sv_cheats 1 para la prueba).\n'
        + ''.join('   exec '+name+'\n' for name in configs) +
        '4. Edita cmd cs_charm_pose en el CFG para ajustar hueso, XYZ, pitch/yaw/roll y escala.\n\n'
        'El VPK contiene solo el charm y sus materiales, sin modelos del arma.\n'
        'Cada arma tiene su CFG y JSON. Los VPK de charm se reutilizan si su geometria y pose coinciden.\n'
        'Cambiar de arma no selecciona automaticamente otro CFG: ejecuta el que corresponda.\n'
        'Reinicia el juego al instalar o actualizar el VPK. Solo un perfil activo.\n'
        'La extension verifica la version exacta de client.dll; no soporta otras versiones ni servidores dedicados.\n'
        'Para detener: script DoEntFire("!self","Kill","",0,null,CSNative.entity)\n'
        'Despues usa plugin_print y plugin_unload con el indice de Charm Studio.\n',encoding='utf8')
    return output
