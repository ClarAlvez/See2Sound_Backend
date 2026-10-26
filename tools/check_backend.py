"""Read-only local checks: no API imports, model loading, downloads or writes."""
from __future__ import annotations

import argparse
import ast
import errno
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROFILE = {
    "SEE2SOUND_APPLY_ACTION_CORRECTIONS": "true",
    "SEE2SOUND_PERSON_INTERMEDIATE": "true",
    "SEE2SOUND_OBJECT_INTERMEDIATE": "true",
    "SEE2SOUND_OBJECT_SUPPRESS_NARRATIVE": "true",
    "SEE2SOUND_OBJECT_APPLY_ADDITIONS": "false",
    "SEE2SOUND_OBJECT_APPLY_REMOVALS": "false",
    "SEE2SOUND_TTS_PROVIDER": "edge",
    "SEE2SOUND_TTS_VOICE": "pt-BR-FranciscaNeural",
    "SEE2SOUND_TTS_RATE": "-5%",
    "SEE2SOUND_TTS_PITCH": "+0Hz",
}
MODULES = (
    "fastapi", "uvicorn", "python_multipart", "pydantic", "cv2", "moviepy",
    "numpy", "faster_whisper", "torch", "torchvision", "transformers", "PIL",
    "scipy", "ultralytics", "imageio_ffmpeg", "llama_cpp", "edge_tts",
    "fvcore", "iopath", "av",
)


def load_settings(root=ROOT):
    # API.core.config creates directories on import. Reuse its Settings definition
    # without executing the module-level instance or mkdir calls.
    path = root / "API/core/config.py"
    tree = ast.parse(path.read_bytes(), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                 or isinstance(node, ast.ClassDef) and node.name == "Settings"]
    namespace = {"__file__": str(path)}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["Settings"]()


def nearest_existing(path):
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def port_available(port):
    """Read-only local connection probe; not a reservation against startup races."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        code = probe.connect_ex(("127.0.0.1", port))
    if code == 0:
        return False
    if code in {errno.ECONNREFUSED, getattr(errno, "WSAECONNREFUSED", 10061)}:
        return True
    raise OSError(code, "Nao foi possivel determinar a disponibilidade da porta local")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-api", action="store_true", help="Nao consultar localhost")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--require-port-free", action="store_true",
                        help="Reprovar se a porta local estiver ocupada antes de iniciar a API")
    args = parser.parse_args(argv)
    if not 1 <= args.api_port <= 65535:
        parser.error("--api-port deve estar entre 1 e 65535")
    failures = 0

    def report(ok, message, *, warning=False):
        nonlocal failures
        label = "OK" if ok else "AVISO" if warning else "ERRO"
        print(f"[{label}] {message}")
        if not ok and not warning:
            failures += 1

    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    print(f"Repositorio: {ROOT}")
    if args.require_port_free:
        try:
            available = port_available(args.api_port)
            report(available, f"Porta local {args.api_port}: " +
                   ("livre neste instante" if available else
                    "ocupada. Verifique a API existente; nenhum processo sera encerrado."))
            if not available:
                return 1
        except OSError as error:
            report(False, f"Verificacao da porta local: {error}")
            return 1
    try:
        settings = load_settings()
    except Exception as error:
        report(False, f"Configuracao invalida: {type(error).__name__}: {error}")
        return 1

    for name, path in settings.get_model_paths().items():
        try:
            valid = path.is_file() and path.stat().st_size > 0
            if valid:
                with path.open("rb") as stream:
                    valid = bool(stream.read(1))
            report(valid, f"Modelo {name}: {path} (presenca/leitura, sem carregar)")
        except OSError as error:
            report(False, f"Modelo {name}: {error}")
    yolo = ROOT / "yolov8n.pt"
    report(yolo.is_file() and os.access(yolo, os.R_OK),
           f"YOLO local: {yolo}; ausente pode exigir download", warning=True)

    for manifest in (ROOT / "requirements.txt", ROOT / "API/requirements.txt"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = re.split(r"[\[<>=!~;\s]", line, maxsplit=1)[0]
            try:
                version = importlib.metadata.version(name)
                exact = re.search(r"==([^;\s]+)", line)
                report(not exact or version == exact.group(1),
                       f"Dependencia {name}: {version} (declarada: {line})")
            except importlib.metadata.PackageNotFoundError:
                report(False, f"Dependencia ausente: {name} ({manifest.name})")
    for module in MODULES:
        report(importlib.util.find_spec(module) is not None, f"Modulo localizavel: {module}")
    print("AVISO: localizar modulos nao comprova imports nativos/DLLs nem compatibilidade entre versoes.")

    for name, expected in PROFILE.items():
        actual = os.getenv(name)
        print(f"[CONFIG] {name}={actual if actual is not None else '(ausente; usa padrao interno)'}"
              f" | demonstracao={expected}")
    print("AVISO: este diagnostico nao ativa nem altera variaveis de ambiente.")

    for path in (settings.runtime_root, settings.upload_dir, settings.jobs_dir, settings.generations_dir):
        anchor = nearest_existing(path)
        report(path.is_dir(), f"Diretorio de saida: {path}", warning=not path.exists())
        report(os.access(anchor, os.R_OK | os.W_OK | os.X_OK),
               f"Permissoes estimadas em {anchor}")
    print("AVISO: permissoes estimadas por os.access; nenhuma escrita de prova foi feita.")
    try:
        disk = shutil.disk_usage(nearest_existing(settings.runtime_root))
        report(disk.free > 0, f"Disco livre: {disk.free / 1024**3:.2f} GiB de {disk.total / 1024**3:.2f} GiB")
        report(disk.free >= 5 * 1024**3, "Margem indicativa de 5 GiB livres; nao garante capacidade do video", warning=True)
    except OSError as error:
        report(False, f"Nao foi possivel consultar disco: {error}")

    if not args.skip_api:
        url = f"http://127.0.0.1:{args.api_port}/health"
        try:
            # Never send this local check through a configured external proxy.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url, timeout=2) as response:
                body = json.loads(response.read(65536))
                ok = response.status == 200 and isinstance(body, dict) and body.get("status") == "ok"
                report(ok, f"API local: {url}; /health comprova apenas resposta HTTP", warning=True)
        except (OSError, ValueError, urllib.error.URLError) as error:
            report(False, f"API local sem resposta valida: {error}; pode estar desligada", warning=True)
    print(f"Diagnostico concluido: {failures} erro(s). Nenhum arquivo foi modificado.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
