"""Versioned script entrypoint shared by the exporter and base packager."""
RUNTIME_PATH = 'charms_studio/runtime_041'
RUNTIME_ENTRY = RUNTIME_PATH + '/start'
RUNTIME_VERSION = '0.4.1-studio'
STUDIO_API = 1
MAX_CONSOLE_BYTES = 240
MAX_CFG_BYTES = 7000


def guarded_command(method, args):
    # The arguments refer to a short, temporary staging table. Keeping both
    # calls separate also fits Source's 512-byte command tokenizer comfortably.
    if method not in ('StudioRegister', 'Studio'):
        raise ValueError('Método de consola no admitido')
    if method == 'StudioRegister':
        ready = '"StudioRegister" in CSCfg.api'
        # StudioRegister returns null on success. If it throws, ok stays false
        # and the next command cannot equip a previously registered wrong model.
        call = f'CSCfg.ok=(CSCfg.api.StudioRegister({args})==null)'
        message = '[Charm Studio] Falta la base Studio 0.4.1'
    else:
        ready = 'CSCfg.ok&&("Studio" in CSCfg.api)'
        call = f'CSCfg.api.Studio({args})'
        message = '[Charm Studio] Perfil cancelado: revisa el error anterior'
    return f'script if({ready}){{{call}}}else{{printl("{message}")}}'
