"""
Quick test to verify code generation fixes
"""

import sys

sys.path.append(".")

from backend.features.code_generator import generate_code

print("Testing code generation with syntax validation...")
print("=" * 80)

# Test query
query = "Create a script to visualize customer behavior"

print(f"Query: {query}")
print("\nGenerating code...")

try:
    code = generate_code(query)

    print("\n" + "=" * 80)
    print("GENERATED CODE:")
    print("=" * 80)
    print(code[:500])  # Print first 500 chars
    print("...")
    print("=" * 80)

    # Try to compile it
    try:
        compile(code, "<test>", "exec")
        print("\n✅ Code compiled successfully!")
        print("   No syntax errors detected.")
    except SyntaxError as e:
        print(f"\n❌ Syntax error in generated code:")
        print(f"   Line {e.lineno}: {e.msg}")
        if e.text:
            print(f"   Text: {e.text[:100]}")

except Exception as e:
    print(f"\n❌ Code generation failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("Test complete")
