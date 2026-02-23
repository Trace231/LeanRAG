"""This module provides the main interfaces for tracing Lean repos, i.e., extracting data from them.
To estimate the time for tracing a repo, a good rule of thumb is 1.5x the time for compiling the repo using :code:`leanpkg build`.
A repo has to be traced only once, and the traced repo will be stored in a cache for fast access in the future.
"""

import itertools
import os
import re
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path
from time import monotonic, sleep
from typing import Generator, List, Optional, Union

from loguru import logger
from tqdm import tqdm

from lean_dojo_v2.utils.common import execute
from lean_dojo_v2.utils.constants import NUM_PROCS
from lean_dojo_v2.utils.filesystem import working_directory

from .cache import cache
from .lean import LeanGitRepo
from .traced_data import TracedRepo

LEAN4_DATA_EXTRACTOR_PATH = Path(__file__).with_name("ExtractData.lean")

_PROGRESSBAR_UPDATE_INTERNAL = 5
_FAILED_PROCESS_PATTERN = re.compile(r"WARNING: Failed to process (?P<path>.+)")


def _read_repo_toolchain(repo_dir: Path) -> Optional[str]:
    toolchain_file = repo_dir / "lean-toolchain"
    if not toolchain_file.exists():
        return None
    raw = toolchain_file.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return raw.splitlines()[0].strip()


def _with_toolchain(cmd: str, toolchain: Optional[str]) -> str:
    if not toolchain:
        return cmd
    return f"elan run {shlex.quote(toolchain)} {cmd}"


def _get_lean_version(toolchain: Optional[str]) -> str:
    output = execute(_with_toolchain("lean --version", toolchain), capture_output=True)[0].strip()
    m = re.match(r"Lean \(version (?P<version>\S+?),", output)
    return m["version"]  # type: ignore[index]


def _modify_dependency_files(packages_path: Path) -> None:
    """Modify dependency files to replace 'import all' with 'public import all'."""
    logger.debug(
        "Modifying dependency files to replace 'import all' with 'public import all'"
    )
    for lean_file in packages_path.rglob("*.lean"):
        with open(lean_file, "r", encoding="utf-8") as f:
            content = f.read()

        modified_content = re.sub(
            r"^(\s*)import all", r"\1public import all", content, flags=re.MULTILINE
        )

        if modified_content != content:
            with open(lean_file, "w", encoding="utf-8") as f:
                f.write(modified_content)


def _sanitize_public_import_all(packages_path: Path) -> None:
    """Normalize invalid `public import all` back to `import all`.

    Some previous runs or legacy logic may rewrite dependency files into a form
    rejected by newer Lean versions. We sanitize before extraction so tracing
    is deterministic across environments.
    """
    logger.debug("Sanitizing dependency files: 'public import all' -> 'import all'")
    for lean_file in packages_path.rglob("*.lean"):
        with open(lean_file, "r", encoding="utf-8") as f:
            content = f.read()

        sanitized = re.sub(
            r"^(\s*)public import all", r"\1import all", content, flags=re.MULTILINE
        )
        if sanitized != content:
            with open(lean_file, "w", encoding="utf-8") as f:
                f.write(sanitized)


def _monitor(paths: List[Path], num_total: int) -> None:
    with tqdm(total=num_total) as pbar:
        while True:
            time_start = monotonic()
            try:
                num_done = len(
                    list(
                        itertools.chain.from_iterable(
                            p.glob(f"**/*.ast.json") for p in paths
                        )
                    )
                )
            except Exception:
                continue
            time_elapsed = monotonic() - time_start
            if time_elapsed < _PROGRESSBAR_UPDATE_INTERNAL:
                sleep(_PROGRESSBAR_UPDATE_INTERNAL - time_elapsed)
            pbar.update(num_done - pbar.n)
            if num_done >= num_total:
                break
    print("")


@contextmanager
def launch_progressbar(paths: List[Path]) -> Generator[None, None, None]:
    """Launch an async progressbar to monitor the progress of tracing the repo."""
    paths = [Path(p) for p in paths]
    olean_files = list(
        itertools.chain.from_iterable(p.glob("**/*.olean") for p in paths)
    )
    num_total = len(olean_files)
    p = Process(target=_monitor, args=(paths, num_total), daemon=True)
    p.start()
    yield
    p.kill()


def get_lean_version() -> str:
    """Get the version of Lean."""
    output = execute("lean --version", capture_output=True)[0].strip()
    m = re.match(r"Lean \(version (?P<version>\S+?),", output)
    return m["version"]  # type: ignore


def is_new_version(v: str) -> bool:
    """Check if ``v`` is at least `4.3.0-rc2`."""
    major, minor, patch = [int(_) for _ in v.split("-")[0].split(".")]
    if major < 4 or (major == 4 and minor < 3):
        return False
    if (
        major > 4
        or (major == 4 and minor > 3)
        or (major == 4 and minor == 3 and patch > 0)
    ):
        return True
    assert major == 4 and minor == 3 and patch == 0
    if "4.3.0-rc" in v:
        rc = int(v.split("-")[1][2:])
        return rc >= 2
    else:
        return True


def check_files(packages_path: Path, no_deps: bool) -> None:
    """Check if all :file:`*.lean` files have been processed to produce :file:`*.ast.json` and :file:`*.dep_paths` files."""
    cwd = Path.cwd()
    packages_path = cwd / packages_path
    jsons = {
        p.with_suffix("").with_suffix("")
        for p in cwd.glob("**/build/ir/**/*.ast.json")
        if not no_deps or not p.is_relative_to(packages_path)
    }
    deps = {
        p.with_suffix("")
        for p in cwd.glob("**/build/ir/**/*.dep_paths")
        if not no_deps or not p.is_relative_to(packages_path)
    }
    oleans = {
        Path(str(p.with_suffix("")).replace("/build/lib/lean/", "/build/ir/"))
        for p in cwd.glob("**/build/lib/lean/**/*.olean")
        if not no_deps or not p.is_relative_to(packages_path)
    }
    assert len(jsons) <= len(oleans) and len(deps) <= len(oleans)
    missing_jsons = {p.with_suffix(".ast.json") for p in oleans - jsons}
    missing_deps = {p.with_suffix(".dep_paths") for p in oleans - deps}
    if len(missing_jsons) > 0 or len(missing_deps) > 0:
        for p in missing_jsons.union(missing_deps):
            logger.warning(f"Missing {p}")


def _extract_failed_paths(stdout: str, stderr: str) -> List[str]:
    failed_paths: List[str] = []
    seen = set()
    for text in (stdout or "", stderr or ""):
        for line in text.splitlines():
            m = _FAILED_PROCESS_PATTERN.search(line.strip())
            if m is None:
                continue
            path = m.group("path").strip()
            if path not in seen:
                seen.add(path)
                failed_paths.append(path)
    return failed_paths


def _repair_failed_files_sequentially(
    failed_paths: List[str], toolchain: Optional[str]
) -> List[str]:
    if not failed_paths:
        return []
    logger.warning(
        "Initial extraction failed for {} files. Starting sequential repair.",
        len(failed_paths),
    )
    still_failed: List[str] = []
    for path in failed_paths:
        try:
            execute(
                _with_toolchain(
                    f"lake env lean --run ExtractData.lean {shlex.quote(path)}",
                    toolchain,
                ),
                capture_output=False,
            )
        except Exception as e:
            logger.warning(f"Sequential repair failed for {path}: {e}")
            still_failed.append(path)
    repaired = len(failed_paths) - len(still_failed)
    logger.info(
        "Sequential repair finished: repaired {} / {} failed files.",
        repaired,
        len(failed_paths),
    )
    return still_failed


def _trace(repo: LeanGitRepo, build_deps: bool) -> None:
    assert (
        repo.exists()
    ), f"The {repo} does not exist. Please check the URL `{repo.commit_url}`."

    # Trace `repo` in the current working directory.
    assert not repo.is_lean4, "Cannot trace Lean 4 itself."
    if not os.path.exists(repo.name):
        repo.clone_and_checkout()
    logger.debug(f"Tracing {repo}")

    with working_directory(repo.name):
        repo_toolchain = _read_repo_toolchain(Path.cwd())
        if repo_toolchain:
            logger.info("Tracing with repo toolchain: {}", repo_toolchain)
        # Build the repo using lake.
        execute(_with_toolchain("lake build", repo_toolchain))

        # Copy the Lean 4 stdlib into the path of packages.
        lean_prefix = execute(
            _with_toolchain("lean --print-prefix", repo_toolchain), capture_output=True
        )[0].strip()
        if is_new_version(_get_lean_version(repo_toolchain)):
            packages_path = Path(".lake/packages")
            build_path = Path(".lake/build")
        else:
            packages_path = Path("lake-packages")
            build_path = Path("build")

        shutil.copytree(
            Path(lean_prefix), str(packages_path / "lean4"), dirs_exist_ok=True
        )

        # Always sanitize dependency files first to avoid parse failures caused
        # by invalid `public import all` directives.
        if build_deps:
            _sanitize_public_import_all(packages_path)

        # NOTE:
        # Rewriting `import all` to `public import all` is incompatible with
        # newer Lean stdlib files and can cause deterministic parse failures
        # ("cannot use `all` with `public import`").
        #
        # Keep this rewrite disabled by default. Enable only if explicitly
        # requested for legacy environments.
        if build_deps and os.getenv("LEAN_DOJO_REWRITE_IMPORT_ALL", "0") == "1":
            logger.warning(
                "Rewriting dependency imports is enabled via "
                "LEAN_DOJO_REWRITE_IMPORT_ALL=1. This may break on newer Lean versions."
            )
            _modify_dependency_files(packages_path)

        # Run ExtractData.lean to extract ASTs, tactic states, and premise information.
        shutil.copyfile(LEAN4_DATA_EXTRACTOR_PATH, LEAN4_DATA_EXTRACTOR_PATH.name)

        dirs_to_monitor = [build_path]
        if build_deps:
            dirs_to_monitor.append(packages_path)
        with launch_progressbar(dirs_to_monitor):
            cmd = f"lake env lean --threads {NUM_PROCS} --run ExtractData.lean"
            if not build_deps:
                cmd += " noDeps"
            full_cmd = _with_toolchain(cmd, repo_toolchain)
            # Do not fail immediately on non-zero exit: ExtractData can fail on a subset
            # of files, and we can recover many of them via sequential repair.
            res = subprocess.run(full_cmd, shell=True, capture_output=True, check=False)
            output = (
                res.stdout.decode(errors="ignore")
                if isinstance(res.stdout, bytes)
                else str(res.stdout or "")
            )
            error = (
                res.stderr.decode(errors="ignore")
                if isinstance(res.stderr, bytes)
                else str(res.stderr or "")
            )
            if res.returncode != 0:
                logger.warning(
                    "ExtractData returned non-zero exit code {}. "
                    "Proceeding with failed-file recovery.",
                    res.returncode,
                )
                stderr_tail = "\n".join((error or "").splitlines()[-40:])
                stdout_tail = "\n".join((output or "").splitlines()[-20:])
                if stderr_tail.strip():
                    logger.warning("ExtractData stderr tail:\n{}", stderr_tail)
                elif stdout_tail.strip():
                    logger.warning(
                        "ExtractData produced no stderr. stdout tail:\n{}",
                        stdout_tail,
                    )
        failed_paths = _extract_failed_paths(output, error)
        still_failed = _repair_failed_files_sequentially(failed_paths, repo_toolchain)
        if still_failed:
            logger.warning(
                "Ignoring {} files that still failed after sequential repair.",
                len(still_failed),
            )

        ast_count = len(list(build_path.glob("**/*.ast.json")))
        if build_deps:
            ast_count += len(list(packages_path.glob("**/*.ast.json")))
        if ast_count == 0:
            raise RuntimeError(
                "ExtractData produced 0 *.ast.json files. "
                "Tracing output is unusable; check the logged ExtractData stderr tail."
            )

        check_files(packages_path, not build_deps)
        os.remove(LEAN4_DATA_EXTRACTOR_PATH.name)


def is_available_in_cache(repo: LeanGitRepo) -> bool:
    """Check if ``repo`` has a traced repo available in the cache (including the remote cache)."""
    rel_cache_dir = repo.get_cache_dirname() / repo.name
    return cache.get(rel_cache_dir) is not None


def get_traced_repo_path(repo: LeanGitRepo, build_deps: bool = True) -> Path:
    """Return the path of a traced repo in the cache.

    The function will trace a repo if it is not available in the cache. See :ref:`caching` for details.

    Args:
        repo (LeanGitRepo): The Lean repo to trace.
        build_deps (bool): Whether to build the dependencies of ``repo``. Defaults to True.

    Returns:
        Path: The path of the traced repo in the cache, e.g. :file:`/home/kaiyu/.cache/lean_dojo/leanprover-community-mathlib-2196ab363eb097c008d4497125e0dde23fb36db2`
    """
    rel_cache_dir = repo.get_cache_dirname() / (
        repo.name + ("_d" if build_deps else "")
    )

    path = cache.get(rel_cache_dir)

    if path is None and not build_deps:
        path = cache.get(rel_cache_dir.parent / (repo.name + "_d"))

    if path is None:
        logger.info(f"Tracing {repo}")
        with working_directory() as tmp_dir:
            logger.debug(f"Working in the temporary directory {tmp_dir}")
            _trace(repo, build_deps)
            src_dir = tmp_dir / repo.name
            traced_repo = TracedRepo.from_traced_files(src_dir, build_deps)
            traced_repo.save_to_disk()
            path = cache.store(src_dir, rel_cache_dir)
    else:
        logger.debug("The traced repo is available in the cache.")
    return path


def trace(
    repo: LeanGitRepo,
    dst_dir: Optional[Union[str, Path]] = None,
    build_deps: bool = True,
) -> TracedRepo:
    """Trace a repo (and its dependencies), saving the results to ``dst_dir``.

    The function only traces the repo when it's not available in the cache. Otherwise,
    it directly copies the traced repo from the cache to ``dst_dir``. See :ref:`caching` for details.

    Args:
        repo (LeanGitRepo): The Lean repo to trace.
        dst_dir (Union[str, Path]): The directory for saving the traced repo. If None, the traced repo is only saved in the cahe.
        build_deps (bool): Whether to build the dependencies of ``repo``. Defaults to True.

    Returns:
        TracedRepo: A :class:`TracedRepo` object corresponding to the files at ``dst_dir``.
    """
    if dst_dir is not None:
        dst_dir = Path(dst_dir)
        assert (
            not dst_dir.exists()
        ), f"The destination directory {dst_dir} already exists."

    rel_cache_dir = repo.get_cache_dirname() / (
        repo.name + ("_d" if build_deps else "")
    )
    cached_path = get_traced_repo_path(repo, build_deps)
    logger.info(f"Loading the traced repo from {cached_path}")
    try:
        traced_repo = TracedRepo.load_from_disk(cached_path, build_deps)
    except RuntimeError as e:
        # Corrupted/incomplete cache can miss git metadata. Recover by rebuilding once.
        if "is not a Git repo" not in str(e):
            raise
        logger.warning(
            "Cached traced repo looks invalid (not a Git repo): {}. Rebuilding cache entry.",
            cached_path,
        )
        if Path(cached_path).exists():
            shutil.rmtree(cached_path, ignore_errors=True)
        with working_directory() as tmp_dir:
            logger.debug(
                "Re-tracing {} in temporary directory {} after cache recovery.",
                repo,
                tmp_dir,
            )
            _trace(repo, build_deps)
            src_dir = tmp_dir / repo.name
            traced_repo = TracedRepo.from_traced_files(src_dir, build_deps)
            traced_repo.save_to_disk()
            cached_path = cache.store(src_dir, rel_cache_dir)
        logger.info("Retry loading rebuilt traced repo from {}", cached_path)
        traced_repo = TracedRepo.load_from_disk(cached_path, build_deps)
    traced_repo.check_sanity()

    if dst_dir is not None:
        dst_dir.mkdir(parents=True)
        shutil.copytree(cached_path, dst_dir / cached_path.name, dirs_exist_ok=True)

    return traced_repo
