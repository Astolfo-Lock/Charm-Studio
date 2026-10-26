# Charm-Studio

Editor visual de charms para Left 4 Dead 2: explora VPK, selecciona huesos y exporta perfiles JSON de posición, rotación y escala.

Versión 1.5.1. Aplicación para Windows con Python 3.10 o superior y PyQt6.

## Uso

Abre los VPK del arma y del charm, elige un hueso y ajusta la transformación. Exporta el perfil JSON para importarlo en ArmsHub.

El flujo recomendado es JSON → ArmsHub. Los exportadores experimentales antiguos permanecen en el código; no se incluye la extensión nativa del juego. La compilación de addons requiere las herramientas de autoría de L4D2.

## Ejecutar desde el código

```powershell
python -m pip install -r requirements.txt
python main.py
```

El ZIP de Releases incluye el programa portable; extrae la carpeta completa antes de abrir el EXE.
No se distribuyen modelos, texturas, addons de terceros ni configuraciones personales.

## Pruebas

```powershell
python -m unittest discover -s tests -v
```

## Crear el ejecutable

```powershell
python -m pip install -r requirements-dev.txt
python tools/build_release.py
```

El proyecto complementario está en https://github.com/Astolfo-Lock/ArmsHub.
