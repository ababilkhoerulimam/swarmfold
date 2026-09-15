import shutil
import stat
import zipfile
from pathlib import Path

from swarmfold.artifacts import sha256_file
from swarmfold.packets import validate_terminal_packet
from swarmfold.schemas import PacketResult

MAX_ARCHIVE_BYTES = 1_000_000_000
MAX_EXTRACTED_BYTES = 2_000_000_000


def create_packet_archive(packet_path: str | Path, archive_path: str | Path) -> str:
    packet_directory = Path(packet_path).resolve()
    validate_terminal_packet(packet_directory)
    output = Path(archive_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(packet_directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("Packet archives cannot contain symbolic links")
            if path.is_file():
                relative = path.relative_to(packet_directory).as_posix()
                archive.write(path, arcname=f"packet/{relative}")
    return sha256_file(output)


def extract_and_validate_archive(archive_path: str | Path, destination: str | Path) -> PacketResult:
    source = Path(archive_path)
    if source.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Packet archive exceeds the upload limit")
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError("Packet archive is empty")
        if sum(member.file_size for member in members) > MAX_EXTRACTED_BYTES:
            raise ValueError("Expanded packet exceeds the extraction limit")
        for member in members:
            parts = Path(member.filename).parts
            if not parts or parts[0] != "packet" or ".." in parts:
                raise ValueError("Packet archive contains an unsafe path")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Packet archive contains a symbolic link")
            target = (output / Path(*parts)).resolve()
            if output != target and output not in target.parents:
                raise ValueError("Packet archive escapes the extraction directory")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source_handle, target.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)
    return validate_terminal_packet(output / "packet")
