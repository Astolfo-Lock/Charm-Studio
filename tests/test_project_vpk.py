import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib
from charmstudio.vpk import VPK, write_vpk
from charmstudio.project import Placement
from charmstudio.resources import Resources
from charmstudio.runtime_contract import MAX_CONSOLE_BYTES, MAX_CFG_BYTES


class ProjectAndArchiveTests(unittest.TestCase):
    def test_split_v2_preload_and_crc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pak01_dir.vpk'
            pre, payload = b'ID', b'STpayload'
            tree = b'mdl\0models/weapons\0v_test\0'
            tree += struct.pack('<IHHIIH', zlib.crc32(pre+payload), len(pre), 0, 3, len(payload), 0xffff)
            tree += pre + b'\0\0\0'
            path.write_bytes(struct.pack('<7I', 0x55AA1234, 2, len(tree), 0, 0, 0, 0) + tree)
            segment = path.with_name('pak01_000.vpk')
            segment.write_bytes(b'pad'+payload)
            reader = VPK(path)
            self.assertEqual(reader.read('MODELS/WEAPONS/V_TEST.MDL'), pre+payload)
            segment.write_bytes(b'pad'+b'badbytes!')
            with self.assertRaisesRegex(ValueError, 'CRC'):
                reader.read('models/weapons/v_test.mdl')

    def test_mount_priority_keeps_selected_workshop_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, workshop = Path(tmp)/'base.vpk', Path(tmp)/'mod.vpk'
            write_vpk(base, {'models/v_test.mdl': b'base', 'models/shared.ani': b'anim'})
            write_vpk(workshop, {'models/v_test.mdl': b'mod'})
            resources = Resources()
            b = resources.mount(base)
            w = resources.mount(workshop)
            self.assertEqual(resources.reader(b)('models/v_test.mdl'), b'base')
            self.assertEqual(resources.reader(w)('models/v_test.mdl'), b'mod')
            self.assertEqual(resources.reader(w)('models/shared.ani'), b'anim')
            with self.assertRaises(ValueError):
                resources.reader(w)('../private')

    def test_profile_roundtrip_and_console_fields_cannot_inject(self):
        record = Placement('models/v_test.mdl', '0'*64, 'models/charm.mdl', '1'*64,
                           'charm;script something()', 'ValveBiped.weapon_bone', [1,2,3], [20,40,60], .5, 'mod')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'p.json'
            record.save(path)
            restored = Placement.load(path)
            self.assertEqual(record, restored)
            commands = restored.commands()
            self.assertNotIn(';', commands)
            self.assertNotIn('something', commands)
            self.assertGreater(len(commands.splitlines()), 3)
            self.assertLessEqual(max(len(line.encode('ascii')) for line in commands.splitlines()), MAX_CONSOLE_BYTES)
            self.assertLessEqual(len(commands.encode('ascii')), MAX_CFG_BYTES)
            record.scale = float('nan')
            with self.assertRaises(ValueError):
                record.commands()
            path.write_text(json.dumps({'schema': 9000}))
            with self.assertRaises((ValueError, TypeError)):
                Placement.load(path)

    def test_archive_paths_reject_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_vpk(Path(tmp)/'bad.vpk', {'../x': b'bad'})

    def test_maximum_length_fields_are_chunked_and_fit_command_buffer(self):
        record = Placement('models/'+'~'*149+'.mdl', '0'*64, 'models/'+'~'*149+'.mdl', '1'*64,
                           '~'*160, '~'*160, [-10000,10000,-10000], [-10000,10000,-10000], .001, '~'*160)
        commands = record.commands()
        self.assertLessEqual(max(map(len,commands.splitlines())),MAX_CONSOLE_BYTES)
        self.assertLessEqual(len(commands),MAX_CFG_BYTES)
        self.assertNotIn(';',commands)
        self.assertNotIn('wait',commands)


if __name__ == '__main__':
    unittest.main()
