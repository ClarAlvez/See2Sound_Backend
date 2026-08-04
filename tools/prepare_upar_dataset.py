import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import gdown
import numpy as np
from tqdm import tqdm


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """

    print(f'* url="{url}"')
    print(f'* destination="{dst}"')

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time + 1e-6
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size + 1e-6)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed" % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    if dst.exists():
        return
    else:
        urllib.request.urlretrieve(url, dst, _reporthook)
        sys.stdout.write("\n")


def extract_zip(src, dst):
    with zipfile.ZipFile(src, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extracting "):
            try:
                zf.extract(member, dst)
            except zipfile.error as err:
                print(err)


def prepare_market(dataset_path):
    # Market 1501 dataset
    market_1501_path = dataset_path / "Market1501"
    market_1501_zipfile = dataset_path / "market_1501.zip"
    url = "https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ"
    print("Download Market 1501 dataset")
    gdown.download(url, output=str(market_1501_zipfile), quiet=False, use_cookies=False)
    print("Extract Market 1501 dataset")
    extract_zip(market_1501_zipfile, dataset_path)
    Path(dataset_path / "Market-1501-v15.09.15").rename(market_1501_path)


def prepare_pa100k(dataset_path):
    # PA-100K dataset
    pa100k_path = dataset_path / "PA100k"
    pa100k_path.mkdir(parents=True, exist_ok=True)
    print("Download PA100k dataset")
    url = "https://drive.google.com/drive/folders/1d_D0Yh7C262gr0ef9EqkvG_M3fqgAWa2?usp=sharing"
    gdown.download_folder(url, output=str(pa100k_path), quiet=False, use_cookies=False)
    print("Extract PA100k dataset")
    extract_zip(pa100k_path / "data.zip", pa100k_path)


def prepare_peta(dataset_path):
    # PETA dataset
    peta_path = dataset_path / "PETA"
    peta_path.mkdir(parents=True, exist_ok=True)

    peta_zipfile = peta_path / "peta.zip"

    print("Download PETA dataset")
    url = "https://www.dropbox.com/s/52ylx522hwbdxz6/PETA.zip?dl=1"
    download_url(url, peta_zipfile)

    print("Extract PETA dataset")
    extract_zip(peta_zipfile, peta_path)

    peta_img_path = peta_path / "images"
    peta_img_path.mkdir(parents=True, exist_ok=True)

    mapping_path = Path("peta_file_mapping.txt")

    if not mapping_path.exists():
        raise FileNotFoundError(
            "Arquivo peta_file_mapping.txt não encontrado na raiz do projeto. "
            "Coloque esse arquivo em See2Sound_Backend/peta_file_mapping.txt"
        )

    mapping = {
        row[0].replace("\\", "/"): row[1].replace("\\", "/")
        for row in np.genfromtxt(mapping_path, dtype=str, delimiter=",")
    }

    moved = 0
    skipped_txt = 0
    skipped_unmapped = 0
    skipped_existing = 0

    files = [
        file
        for file in peta_path.rglob("*")
        if file.is_file()
    ]

    for file in tqdm(files, desc="Organizing PETA"):
        if file.suffix.lower() == ".txt":
            skipped_txt += 1
            continue

        if file.name == "peta.zip":
            continue

        try:
            relative_to_dataset = file.relative_to(dataset_path).as_posix()
        except ValueError:
            relative_to_dataset = file.as_posix()

        try:
            relative_to_peta = file.relative_to(peta_path).as_posix()
        except ValueError:
            relative_to_peta = file.as_posix()

        candidate_keys = [
            relative_to_dataset,
            relative_to_peta,
            str(PurePosixPath(file)).replace("\\", "/"),
        ]

        destination_relative = None

        for key in candidate_keys:
            if key in mapping:
                destination_relative = mapping[key]
                break

        if destination_relative is None:
            skipped_unmapped += 1
            continue

        destination = dataset_path / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            skipped_existing += 1
            continue

        shutil.move(str(file), str(destination))
        moved += 1

    print("\nPETA organizada.")
    print("Arquivos movidos:", moved)
    print("TXT ignorados:", skipped_txt)
    print("Arquivos sem mapping:", skipped_unmapped)
    print("Arquivos já existentes:", skipped_existing)

def prepare_annotations(dataset_path):
    anno_path = dataset_path / "annotations"
    anno_path.mkdir(parents=True, exist_ok=True)
    anno_zipfile = anno_path / "development.zip"
    print("Download annotations")
    url = "https://drive.google.com/file/d/1FMX9nUrXArxW4wkORO6Z7zp7xy7JBjUM/view?usp=sharing"
    gdown.download(url, output=str(anno_zipfile), quiet=False, use_cookies=False)
    print("Extract annotations")
    extract_zip(anno_zipfile, anno_path)


def prepare_templates(dataset_path):
    template_zipfile = dataset_path / "submission_templates.zip"
    print("Download templates")
    url = "https://drive.google.com/file/d/11ZxT8kixkV-vAj8aixS8n2aGJ5Rw0OQy/view?usp=sharing"
    gdown.download(url, output=str(template_zipfile), quiet=False, use_cookies=False)
    print("Extract templates")
    extract_zip(template_zipfile, dataset_path)


def prepare_datasets(path):
    # Datasets folder
    dataset_path = Path(path)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Já foram baixados/extraídos. Vamos apenas corrigir PETA e continuar.
    prepare_peta(dataset_path)

    # Download & extract annotations
    prepare_annotations(dataset_path)

    # Download & extract submission templates
    prepare_templates(Path("./"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WACV2024 RWS UPAR Challenge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Dataset directory. Downloaded datasets are stored in this directory.",
    )
    args = parser.parse_args()

    prepare_datasets(args.data_dir)