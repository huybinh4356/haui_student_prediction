import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_JVM_STARTED = False
_H2_JAR_PATH: Optional[str] = None


def _find_h2_jar() -> Optional[str]:
    env_jar = os.getenv("HAUI_H2_JAR")
    if env_jar and Path(env_jar).exists():
        return str(Path(env_jar).resolve())
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent / "vendor" / "h2" / "h2-2.5.252.jar",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _discover_java_home() -> Optional[str]:
    jh = os.getenv("JAVA_HOME")
    if jh and Path(jh).exists():
        return jh
    try:
        import subprocess
        result = subprocess.run(["java", "-XshowSettings:properties", "-version"],
                                capture_output=True, text=True, timeout=10)
        for line in (result.stdout + result.stderr).splitlines():
            if "java.home" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    return parts[1].strip()
    except Exception:
        pass
    return None


def start_jvm() -> None:
    global _JVM_STARTED, _H2_JAR_PATH
    if _JVM_STARTED:
        return
    try:
        import jpype
    except ImportError as e:
        raise RuntimeError("JPype1 is not installed. Install it: pip install JPype1") from e

    jar = _find_h2_jar()
    if not jar:
        raise RuntimeError(
            "H2 JDBC JAR not found. "
            "Set HAUI_H2_JAR env var or place jar at vendor/h2/h2-2.5.252.jar"
        )
    _H2_JAR_PATH = jar

    java_home = _discover_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home

    jpype.startJVM(classpath=[jar])
    _JVM_STARTED = True
    logger.info("JVM started with H2 JAR: %s", jar)


def get_connection(url: Optional[str] = None):
    start_jvm()
    import jpype
    jdbc_url = url or os.getenv("HAUI_H2_URL", "jdbc:h2:mem:haui_session;DB_CLOSE_DELAY=-1")
    return jpype.java.sql.DriverManager.getConnection(jdbc_url, "sa", "")


def shutdown_jvm() -> None:
    global _JVM_STARTED
    if _JVM_STARTED:
        import jpype
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
        _JVM_STARTED = False
        logger.info("JVM shutdown")


def init_schema(conn) -> None:
    sql_path = Path(__file__).with_name("schema.sql")
    if not sql_path.exists():
        raise FileNotFoundError(f"Schema file not found: {sql_path}")
    script = sql_path.read_text(encoding="utf-8")
    stmt = conn.createStatement()
    try:
        for line in script.split(";"):
            line = line.strip()
            if line and not line.startswith("--"):
                stmt.execute(line)
    finally:
        stmt.close()
