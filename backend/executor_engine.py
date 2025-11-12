# backend/executor_engine.py
import os
import json
import sqlite3
import traceback
import signal
import io
import sys
import base64
import subprocess
from datetime import datetime
from pathlib import Path
from backend.features.summariser import Summariser
import config  # Import config to set matplotlib backend


class TimeoutError(Exception):
    """Raised when code execution times out."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Code execution timed out")


class ExecutorEngine:
    def __init__(self, db_path="data/olist.db"):
        self.db_path = db_path
        self.summariser = Summariser()
        self._conn = None

    def _get_connection(self):
        """Get or create database connection with connection reuse."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def _install_missing_library(self, library_name, package_name=None):
        """Install a missing library using pip."""
        if package_name is None:
            package_name = library_name

        try:
            # Use subprocess to run pip install
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return True, f"Successfully installed {package_name}"
            else:
                return False, f"Failed to install {package_name}: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, f"Timeout installing {package_name}"
        except Exception as e:
            return False, f"Error installing {package_name}: {str(e)}"

    def _detect_and_install_libraries(self, code):
        """Detect import statements in code and install missing libraries."""
        import re

        import_statements = []

        # Find import statements
        import_patterns = [
            r"^import\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"^from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import",
        ]

        for line in code.split("\n"):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    lib = match.group(1)
                    if (
                        lib not in ["os", "sys", "json", "sqlite3", "pandas", "numpy"]
                        and lib not in import_statements
                    ):
                        import_statements.append(lib)

        # Map common libraries to pip packages
        library_map = {
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "plotly": "plotly",
            "sklearn": "scikit-learn",
            "PIL": "pillow",
            "cv2": "opencv-python",
            "tensorflow": "tensorflow",
            "torch": "torch",
            "scipy": "scipy",
            "requests": "requests",
            "beautifulsoup4": "beautifulsoup4",
            "lxml": "lxml",
        }

        installation_results = []
        for lib in import_statements:
            package = library_map.get(lib, lib)
            try:
                __import__(lib)
                installation_results.append(f"✓ {lib} already available")
            except ImportError:
                success, message = self._install_missing_library(lib, package)
                if success:
                    installation_results.append(f"✓ Installed {lib}")
                else:
                    installation_results.append(f"✗ Failed to install {lib}: {message}")

        return installation_results

    # -------------------------------
    # SQL EXECUTION
    # -------------------------------
    def execute_sql(self, query: str):
        result = {
            "status": "failed",
            "rows": 0,
            "columns": [],
            "data": [],
            "summary": "",
        }
        try:
            # Basic SQL safety checks
            query_upper = query.upper().strip()

            # Only allow SELECT statements
            if not query_upper.startswith("SELECT"):
                result["error"] = "Only SELECT queries are allowed"
                return result

            # Block dangerous operations
            dangerous_keywords = [
                "DROP",
                "DELETE",
                "INSERT",
                "UPDATE",
                "ALTER",
                "CREATE",
                "TRUNCATE",
                "EXEC",
                "EXECUTE",
            ]
            if any(keyword in query_upper for keyword in dangerous_keywords):
                result["error"] = "Query contains forbidden operations"
                return result

            # Limit query length
            if len(query) > 5000:
                result["error"] = "Query too long (max 5000 characters)"
                return result

            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []

            result["status"] = "success"
            result["rows"] = len(rows)
            result["columns"] = columns
            result["data"] = rows
            result["summary"] = self.summariser.summarise_sql_result(rows, columns)
        except Exception as e:
            result["error"] = str(e)
            traceback.print_exc()
        return result

    # -------------------------------
    # PYTHON CODE EXECUTION
    # -------------------------------
    def execute_python(self, filepath: str):
        result = {
            "status": "failed",
            "stdout": "",
            "error": "",
            "output_type": "text",  # text, image, or mixed
            "images": [],  # list of base64 encoded images
            "has_visualization": False,
        }
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()

            # Validate code before execution
            code = code.strip()
            if not code:
                result["error"] = "Empty code file"
                return result

            # Check for common LLM artifacts that shouldn't be in code
            lines = code.split("\n")
            first_line = lines[0].strip()

            # Remove any leading non-Python text (common LLM error)
            if first_line and not any(
                first_line.startswith(kw)
                for kw in [
                    "import",
                    "from",
                    "#",
                    "def",
                    "class",
                    "if",
                    "try",
                    "with",
                    "for",
                    "while",
                    "@",
                    "async",
                    "print(",
                    "raise",
                    "return",
                    "yield",
                    "global",
                    "nonlocal",
                    "assert",
                    "break",
                    "continue",
                    "pass",
                ]
            ):
                # Invalid first line, try to find first valid line
                valid_start = 0
                for i, line in enumerate(lines):
                    line_stripped = line.strip()
                    if line_stripped and any(
                        line_stripped.startswith(kw)
                        for kw in [
                            "import",
                            "from",
                            "#",
                            "def",
                            "class",
                            "if",
                            "try",
                            "with",
                            "for",
                            "while",
                            "print(",
                            "@",
                        ]
                    ):
                        valid_start = i
                        break

                if valid_start > 0:
                    code = "\n".join(lines[valid_start:])
                    result[
                        "stdout"
                    ] += (
                        f"⚠️ Removed {valid_start} invalid line(s) from generated code\n"
                    )
                else:
                    result["error"] = (
                        f"Invalid Python code: starts with '{first_line[:50]}...'"
                    )
                    return result

            # Check if code contains visualization libraries
            has_matplotlib = (
                "matplotlib" in code
                or "plt." in code
                or "plt.show" in code
                or "pyplot" in code
                or "import matplotlib" in code
            )
            has_seaborn = (
                "seaborn" in code or "sns." in code or "import seaborn" in code
            )
            has_plotly = (
                "plotly" in code
                or "px." in code
                or "go." in code
                or "import plotly" in code
                or "plotly.graph_objects" in code
            )

            result["has_visualization"] = has_matplotlib or has_seaborn or has_plotly

            # Capture stdout
            old_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output

            try:
                # Detect and install any missing libraries from the code
                install_results = self._detect_and_install_libraries(code)
                if install_results:
                    result["stdout"] += "\n".join(install_results) + "\n"

                if result["has_visualization"]:

                    # Remove plt.show() calls since we're capturing figures
                    code = code.replace(
                        "plt.show()", "# plt.show() - removed for capture"
                    )

                    # Set matplotlib backend BEFORE any matplotlib imports in user code
                    import matplotlib  # type: ignore

                    matplotlib.use("Agg")  # Use non-interactive backend

                    # Define execute_query helper function for generated code
                    def execute_query(query):
                        """Helper function for executing SQL queries in generated code"""
                        import sqlite3
                        import pandas as pd

                        conn = sqlite3.connect("data/olist.db")
                        try:
                            df = pd.read_sql_query(query, conn)
                            return df
                        finally:
                            conn.close()

                    # For visualization code, use full environment but capture plots
                    exec_globals = {
                        "__builtins__": __builtins__,
                        "__name__": "__main__",  # Allow if __name__ == '__main__' to work
                        "pd": __import__("pandas"),
                        "np": __import__("numpy"),
                        "sqlite3": __import__("sqlite3"),
                        "os": __import__("os"),
                        "sys": __import__("sys"),
                        "json": __import__("json"),
                        "execute_query": execute_query,  # Add helper function
                    }

                    # Try to import visualization libraries
                    try:
                        import matplotlib.pyplot as plt  # type: ignore

                        exec_globals["plt"] = plt
                        exec_globals["matplotlib"] = matplotlib
                    except ImportError as e:
                        result[
                            "error"
                        ] += f"Warning: matplotlib not available: {str(e)}\n"

                    try:
                        import seaborn as sns  # type: ignore

                        exec_globals["sns"] = sns
                    except ImportError as e:
                        result["error"] += f"Warning: seaborn not available: {str(e)}\n"

                    try:
                        import plotly.express as px  # type: ignore
                        import plotly.graph_objects as go  # type: ignore

                        exec_globals["px"] = px
                        exec_globals["go"] = go
                    except ImportError as e:
                        result["error"] += f"Warning: plotly not available: {str(e)}\n"

                    # Add warnings module for generated code that uses it
                    try:
                        import warnings

                        exec_globals["warnings"] = warnings
                    except ImportError:
                        pass

                    exec_locals = {}

                    # Execute the code
                    exec(code, exec_globals, exec_locals)

                    # Check for matplotlib figures and save them
                    try:
                        import matplotlib.pyplot as plt  # type: ignore

                        figures = plt.get_fignums()
                        if figures:
                            result["output_type"] = "image"
                            for fig_num in figures:
                                try:
                                    fig = plt.figure(fig_num)
                                    buf = io.BytesIO()
                                    fig.savefig(
                                        buf, format="png", dpi=100, bbox_inches="tight"
                                    )
                                    buf.seek(0)
                                    img_base64 = base64.b64encode(
                                        buf.getvalue()
                                    ).decode("utf-8")
                                    result["images"].append(
                                        f"data:image/png;base64,{img_base64}"
                                    )
                                    plt.close(fig)
                                except Exception as fig_error:
                                    result[
                                        "error"
                                    ] += f"Error saving figure {fig_num}: {str(fig_error)}\n"
                            plt.close("all")  # Close all figures to free memory
                    except Exception as e:
                        result[
                            "error"
                        ] += f"Error processing matplotlib figures: {str(e)}\n"

                else:
                    # For non-visualization code, use restricted environment
                    safe_builtins = {
                        "__import__": __import__,
                        "print": print,
                        "len": len,
                        "sum": sum,
                        "range": range,
                        "int": int,
                        "float": float,
                        "str": str,
                        "list": list,
                        "dict": dict,
                        "tuple": tuple,
                        "set": set,
                        "bool": bool,
                        "abs": abs,
                        "min": min,
                        "max": max,
                        "round": round,
                        "sorted": sorted,
                        "enumerate": enumerate,
                        "zip": zip,
                        "map": map,
                        "filter": filter,
                    }

                    exec_globals = {
                        "__builtins__": safe_builtins,
                        "pd": __import__("pandas"),
                        "np": __import__("numpy"),
                    }
                    exec_locals = {}

                    exec(code, exec_globals, exec_locals)

                # Get captured output
                result["stdout"] = captured_output.getvalue()
                result["status"] = "success"

            except Exception as e:
                result["error"] = str(e)
                result["stdout"] = captured_output.getvalue()
                traceback.print_exc()
            finally:
                sys.stdout = old_stdout

        except Exception as e:
            result["error"] = str(e)
            traceback.print_exc()

        return result

    # -------------------------------
    # SAVE EXECUTION LOG
    # -------------------------------
    def log_execution(self, query_type, query, result):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"execution_{timestamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"type": query_type, "query": query, "result": result}, f, indent=2
            )
        return path
