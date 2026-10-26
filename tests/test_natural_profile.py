import unittest

from charmstudio.natural_profile import (arms_entries, patch_arms, find_weapon_script,
                                        resources_with_game_fallback)


class NaturalScriptTests(unittest.TestCase):
    def test_workshop_only_automatically_resolves_official_scripts_and_arms(self):
        from pathlib import Path
        import tempfile
        from charmstudio.resources import Resources
        from charmstudio.vpk import write_vpk
        script = (b'WeaponData { "viewmodel" "models/v_models/v_rifle_ak47.mdl" '
                  b'"CharacterViewmodelAddon" { "Gambler" "models/arms/nick.mdl" } }')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'left4dead2').mkdir()
            (root/'update').mkdir()
            write_vpk(root/'left4dead2/pak01_dir.vpk', {
                'scripts/weapon_rifle_ak47.txt': script+b' // old base',
                'models/arms/nick.mdl': b'official arms'})
            write_vpk(root/'update/pak01_dir.vpk', {
                'scripts/weapon_rifle_ak47.txt': script+b' // current update'})
            write_vpk(root/'workshop.vpk', {
                'models/v_models/v_rifle_ak47.mdl': b'workshop weapon'})
            resources = Resources()
            preferred = resources.mount(root/'workshop.vpk')
            with self.assertRaisesRegex(ValueError, 'No hay scripts'):
                find_weapon_script(resources, preferred, 'models/v_models/v_rifle_ak47.mdl')
            resolved = resources_with_game_fallback(resources, root)
            path, data, mount = find_weapon_script(resolved, preferred, 'models/v_models/v_rifle_ak47.mdl')
            self.assertEqual(path, 'scripts/weapon_rifle_ak47.txt')
            self.assertTrue(data.endswith(b'current update'))
            self.assertEqual(mount.path, (root/'update/pak01_dir.vpk').resolve())
            self.assertEqual(resolved.reader()('models/arms/nick.mdl'), b'official arms')
            self.assertEqual(resolved.reader()('models/v_models/v_rifle_ak47.mdl'), b'workshop weapon')
            self.assertEqual(resources.mounts, [preferred], 'Export must not modify UI mounts')
            # A Workshop script override remains authoritative, including custom paths.
            write_vpk(root/'override.vpk', {'scripts/weapon_rifle_ak47.txt':
                script.replace(b'models/v_models/v_rifle_ak47.mdl', b'models/custom/ak.mdl')})
            override = resources.mount(root/'override.vpk')
            resolved = resources_with_game_fallback(resources, root)
            self.assertIs(find_weapon_script(resolved, override, 'models/custom/ak.mdl')[2], override)
            with self.assertRaisesRegex(ValueError, 'Ningún script'):
                find_weapon_script(resolved, override, 'models/v_models/v_rifle_ak47.mdl')

    def test_script_resolution_uses_model_and_addon_priority_for_multiple_weapons(self):
        from types import SimpleNamespace
        def script(model, damage):
            return (f'WeaponData {{ "viewmodel" "{model}" "Damage" "{damage}" '
                    '"CharacterViewmodelAddon" { "Gambler" "models/arms/nick.mdl" } }').encode()
        official = {'scripts/weapon_rifle.txt': script('models/v_models/v_rifle.mdl', 33),
                    'scripts/weapon_rifle_ak47.txt': script('models/v_models/v_rifle_ak47.mdl', 58)}
        addon = {'scripts/weapon_rifle.txt': script('models/custom/rifle.mdl', 99)}
        base = SimpleNamespace(files=official, read=official.__getitem__)
        mod = SimpleNamespace(files=addon, read=addon.__getitem__)
        resources = SimpleNamespace(mounts=[base, mod])
        path, data, source = find_weapon_script(resources, mod, 'models/custom/rifle.mdl')
        self.assertEqual(path, 'scripts/weapon_rifle.txt')
        self.assertIs(source, mod)
        self.assertIn(b'"Damage" "99"', data)
        path, _, source = find_weapon_script(resources, mod, 'models/v_models/v_rifle_ak47.mdl')
        self.assertEqual(path, 'scripts/weapon_rifle_ak47.txt')
        self.assertIs(source, base)
        with self.assertRaises(ValueError):
            find_weapon_script(resources, mod, 'models/v_models/v_rifle.mdl')

    def test_only_model_values_change_in_table(self):
        data = (b'WeaponData\r\n{\r\n// "CharacterViewmodelAddon" { ignored }\r\n'
                b'"Damage" "58" "viewmodel" "models/v_models/v_rifle.mdl"\r\n'
                b'"CharacterViewmodelAddon" { "Gambler" "models/arms/nick.mdl" '
                b'// keep comment\r\n"Coach" "models/arms/coach.mdl" }\r\n'
                b'"Other" { "Coach" "unchanged" } }')
        replacements = {'Gambler': 'models/charms_studio/natural/n_nick.mdl',
                        'Coach': 'models/charms_studio/natural/n_coach.mdl'}
        actual = patch_arms(data, replacements)
        expected = data.replace(b'models/arms/nick.mdl', replacements['Gambler'].encode()).replace(
            b'models/arms/coach.mdl', replacements['Coach'].encode())
        self.assertEqual(actual, expected)
        self.assertEqual(len(arms_entries(actual)), 2)

    def test_reject_incomplete_or_ambiguous_tables(self):
        with self.assertRaises(ValueError):
            patch_arms(b'"CharacterViewmodelAddon" { "Gambler" "models/a.mdl" }', {})
        for data in [b'"CharacterViewmodelAddon" { "Gambler" "models/a.mdl"',
                     b'"CharacterViewmodelAddon" { "Gambler" "models/a.mdl" "gambler" "models/b.mdl" }',
                     b'"CharacterViewmodelAddon" { "Gambler" "models/a.mdl" } "CharacterViewmodelAddon" {}']:
            with self.assertRaises(ValueError):
                arms_entries(data)


if __name__ == '__main__':
    unittest.main()
