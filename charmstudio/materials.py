"""Repairs for the two supplied charm packs; source files are never changed."""
import re


def repair_material(path, content, read):
    """Return (VMT bytes, repair notes). Only known supplied-pack mistakes change.

    An unavailable detail layer is cosmetic and can be omitted. Main textures
    remain required by the packager. Reader may include mounted game resources.
    """
    if not path.lower().endswith('.vmt'):
        return content, []
    text = content.decode('utf8', errors='replace')
    notes = []
    def exists(resource):
        try:
            read(resource)
            return True
        except (FileNotFoundError, KeyError):
            return False
    normalized = path.replace('\\', '/').lower()
    correction = {
        'materials/models/astolfo_charm/metalwall076a.vmt': 'models/astolfo_charm/metalwall076a',
        'materials/models/bo3/metalwall005b.vmt': 'models/bo3/bo3_k',
    }.get(normalized)
    base = re.compile(r'(?im)("?\$basetexture"?\s+)(?:"([^"\r\n]+)"|([^\s{}]+))')
    match = base.search(text)
    if correction and match:
        old = (match.group(2) or match.group(3)).replace('\\', '/')
        if old.lower() == 'models/astolfo_charm/metalwall076' and not exists('materials/'+old+'.vtf') and exists('materials/'+correction+'.vtf'):
            text = base.sub(lambda m: m.group(1)+'"'+correction+'"', text, count=1)
            notes.append(f'{normalized}: $basetexture {old} -> {correction}')
    detail = re.compile(r'(?im)^[ \t]*"?\$detail"?\s+(?:"([^"\r\n]+)"|([^\s{}]+))[^\r\n]*')
    def omit_missing(match):
        name = (match.group(1) or match.group(2)).replace('\\', '/')
        if not exists('materials/'+name.removeprefix('materials/').removesuffix('.vtf')+'.vtf'):
            notes.append(f'{normalized}: omitida capa $detail ausente: {name}')
            return '// Charm Studio: optional detail texture unavailable'
        return match.group(0)
    text = detail.sub(omit_missing, text)
    return text.encode('utf8'), notes
