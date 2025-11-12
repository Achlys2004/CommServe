"""
Code validation and fixing utilities
"""

import re
import ast
import logging

logger = logging.getLogger(__name__)


def balance_quotes(code: str) -> str:
    """
    Attempt to balance unmatched quotes and braces in code.
    This is a heuristic approach to fix common LLM code generation issues.
    """
    lines = code.split("\n")
    fixed_lines = []

    for line in lines:
        # Skip comments and docstrings
        stripped = line.strip()
        if stripped.startswith("#"):
            fixed_lines.append(line)
            continue

        # Count quotes
        single_quotes = line.count("'") - line.count("\\'")
        double_quotes = line.count('"') - line.count('\\"')

        # Count braces (for f-strings)
        open_braces = line.count("{") - line.count("}")
        close_braces = line.count("}") - line.count("{")

        # If odd number of quotes, try to fix
        if single_quotes % 2 == 1:
            # Unterminated single quote - add one at the end
            if "'" in line and not line.rstrip().endswith("'"):
                line = line.rstrip() + "'"
                logger.debug(f"Added closing single quote to: {line[:50]}...")

        if double_quotes % 2 == 1:
            # Unterminated double quote - add one at the end
            if '"' in line and not line.rstrip().endswith('"'):
                line = line.rstrip() + '"'
                logger.debug(f"Added closing double quote to: {line[:50]}...")

        # Fix unbalanced braces in f-strings
        if open_braces > 0 and 'f"' in line or "f'" in line:
            # This is likely an f-string with missing closing brace
            line = line.rstrip() + "}" * open_braces
            logger.debug(f"Added closing braces to f-string: {line[:50]}...")

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def remove_incomplete_lines(code: str) -> str:
    """
    Remove lines that appear incomplete (e.g., function calls without closing parens)
    """
    lines = code.split("\n")
    cleaned_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            cleaned_lines.append(line)
            continue

        # Check for lines that require indentation but are followed by unindented content
        if stripped.endswith(":"):
            # This line starts a block, check next line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.strip()

                # If next line exists but isn't indented and isn't empty/comment
                if next_stripped and not next_stripped.startswith("#"):
                    # Check indentation
                    current_indent = len(line) - len(line.lstrip())
                    next_indent = len(next_line) - len(next_line.lstrip())

                    if next_indent <= current_indent:
                        # Next line not indented enough, add a pass statement
                        indent_spaces = " " * (current_indent + 4)
                        cleaned_lines.append(line)
                        cleaned_lines.append(f"{indent_spaces}pass")
                        logger.debug(f"Added 'pass' after: {line[:50]}...")
                        continue

        # Check for obvious incomplete lines
        open_parens = line.count("(") - line.count(")")
        open_brackets = line.count("[") - line.count("]")
        open_braces = line.count("{") - line.count("}")

        # If line has unclosed brackets and is not followed by continuation
        if open_parens > 0 or open_brackets > 0 or open_braces > 0:
            # Check if next line continues this statement
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not next_line.startswith("#"):
                    # Likely a multi-line statement, keep it
                    cleaned_lines.append(line)
                    continue

            # Line appears incomplete, try to close it
            if open_parens > 0:
                line = line.rstrip() + ")" * open_parens
                logger.debug(f"Closed parentheses in: {line[:50]}...")
            if open_brackets > 0:
                line = line.rstrip() + "]" * open_brackets
            if open_braces > 0:
                line = line.rstrip() + "}" * open_braces

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def fix_incomplete_try_blocks(code: str) -> str:
    """
    Fix incomplete try blocks by adding missing except clauses.
    """
    lines = code.split("\n")
    fixed_lines = []
    in_try_block = False
    try_indent = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("try:"):
            in_try_block = True
            try_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue

        if in_try_block:
            current_indent = len(line) - len(line.lstrip())

            # If we find a line with same or less indentation than try, and no except/finally yet
            if (
                current_indent <= try_indent
                and stripped
                and not stripped.startswith("#")
            ):
                if not stripped.startswith(("except", "finally", "else:")):
                    # Add a generic except block before this line
                    indent_spaces = " " * try_indent
                    fixed_lines.append(f"{indent_spaces}except Exception as e:")
                    fixed_lines.append(f'{indent_spaces}    print(f"Error: {{e}}")')
                    fixed_lines.append(f"{indent_spaces}    pass")
                    in_try_block = False

        if stripped.startswith(("except", "finally")):
            in_try_block = False

        fixed_lines.append(line)

    # If still in try block at end, add except
    if in_try_block:
        indent_spaces = " " * try_indent
        fixed_lines.append(f"{indent_spaces}except Exception as e:")
        fixed_lines.append(f'{indent_spaces}    print(f"Error: {{e}}")')
        fixed_lines.append(f"{indent_spaces}    pass")

    return "\n".join(fixed_lines)


def fix_fstring_issues(code: str) -> str:
    """
    Fix common f-string issues like multiline expressions.
    """
    lines = code.split("\n")
    fixed_lines = []
    in_fstring = False
    fstring_start = ""

    for line in lines:
        stripped = line.strip()

        # Check if this line starts an f-string
        if ('f"' in line or "f'" in line) and not in_fstring:
            # Count quotes to see if f-string is properly closed on same line
            if ('f"' in line and line.count('"') % 2 == 0) or (
                "f'" in line and line.count("'") % 2 == 0
            ):
                # F-string is properly closed on same line
                fixed_lines.append(line)
            else:
                # F-string spans multiple lines
                in_fstring = True
                fstring_start = line
                continue
        elif in_fstring:
            # We're in a multiline f-string, look for the closing quote
            if ('"' in line and fstring_start.count('"') > 0) or (
                "'" in line and fstring_start.count("'") > 0
            ):
                # Found closing quote, combine the lines
                combined = fstring_start.rstrip() + " " + line.lstrip()
                fixed_lines.append(combined)
                in_fstring = False
            else:
                # Still in f-string, combine with previous
                fstring_start = fstring_start.rstrip() + " " + line.lstrip()
        else:
            fixed_lines.append(line)

    # If still in f-string at end, add closing quote
    if in_fstring:
        if 'f"' in fstring_start:
            fixed_lines.append(fstring_start + '"')
        elif "f'" in fstring_start:
            fixed_lines.append(fstring_start + "'")

    return "\n".join(fixed_lines)


def validate_and_fix_code(code: str, max_fixes: int = 3) -> tuple[bool, str, str]:
    """
    Validate Python code and attempt automatic fixes for common issues.

    Returns:
        (is_valid, fixed_code, error_message)
    """
    original_code = code
    attempt = 0

    while attempt < max_fixes:
        try:
            # Try to compile the code
            compile(code, "<string>", "exec")

            # Also try AST parsing for better validation
            ast.parse(code)

            # Success!
            if attempt > 0:
                logger.info(f"Code fixed after {attempt} attempts")
            return True, code, ""

        except SyntaxError as e:
            attempt += 1
            logger.warning(
                f"Syntax error (attempt {attempt}/{max_fixes}): {e.msg} at line {e.lineno}"
            )

            if (
                "unterminated string" in str(e).lower()
                or "f-string: expecting" in str(e).lower()
            ):
                # Try to fix unterminated strings and f-string issues
                code = balance_quotes(code)
                continue

            elif (
                "unexpected EOF" in str(e).lower() or "invalid syntax" in str(e).lower()
            ):
                # Try to fix incomplete lines
                code = remove_incomplete_lines(code)
                continue

            elif "expected an indented block" in str(e).lower():
                # Try to fix indentation issues
                code = remove_incomplete_lines(code)
                continue

            elif "expected 'except' or 'finally' block" in str(e).lower():
                # Try to fix incomplete try blocks
                code = fix_incomplete_try_blocks(code)
                continue

            elif "f-string" in str(e).lower() or "expecting" in str(e).lower():
                # Try to fix f-string issues
                code = fix_fstring_issues(code)
                continue  # Can't automatically fix this error
            return False, original_code, f"{e.msg} at line {e.lineno}"

        except Exception as e:
            # Other parsing errors
            return False, original_code, str(e)

    # Max attempts reached without success
    return False, original_code, f"Could not fix code after {max_fixes} attempts"


def extract_python_code(text: str) -> str:
    """
    Extract Python code from text that may contain markdown or other formatting.
    More robust than simple string splitting.
    """
    # Try to find code blocks
    if "```python" in text:
        # Find all python code blocks
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

    if "```py" in text:
        pattern = r"```py\s*(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

    if "```" in text:
        # Generic code block
        pattern = r"```\s*(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return first block that looks like Python
            for match in matches:
                if "import" in match or "def " in match or "print(" in match:
                    return match.strip()
            # Fallback to first block
            return matches[0].strip()

    # No code blocks found, return as-is
    return text.strip()
