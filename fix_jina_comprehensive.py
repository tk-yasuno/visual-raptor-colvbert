#!/usr/bin/env python3
"""
Fix jina_vdr_benchmark.py - comprehensive fix
"""

import pathlib
import subprocess
import re

# ファイルを読み込む
file_path = pathlib.Path('jina_vdr_benchmark.py')
print(f"Reading {file_path}...")

content = file_path.read_text(encoding='utf-8')
print(f"Original file size: {len(content)} characters")

# 問題のあるパターンを検出
escaped_newlines = content.count('\\n')
escaped_backslashes = len(re.findall(r'= \\\\$', content, re.MULTILINE))

print(f"Found {escaped_newlines} escaped newlines (\\n)")
print(f"Found {escaped_backslashes} line continuations (\\\\)")

# 修正を適用
fixed_content = content

# 1. エスケープされた\nを実際の改行に置換
fixed_content = fixed_content.replace('\\n', '\n')

# 2. 行末の\\ を削除（Python では不要な行継続）
fixed_content = re.sub(r' \\\\$', '', fixed_content, flags=re.MULTILINE)

print(f"Fixed file size: {len(fixed_content)} characters")

# バックアップを作成
backup_path = file_path.with_name(file_path.stem + '_backup' + file_path.suffix)
if file_path.exists():
    import shutil
    shutil.copy2(file_path, backup_path)
    print(f"Backup created: {backup_path}")

# 修正したファイルを書き込む
file_path.write_text(fixed_content, encoding='utf-8')
print(f"Fixed file written: {file_path}")

# 構文チェック
print("\nRunning syntax check...")
result = subprocess.run(['python', '-m', 'py_compile', str(file_path)], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Syntax check passed!")
    print("\nSummary:")
    print(f"- Fixed {escaped_newlines} escaped newlines")
    print(f"- Removed {escaped_backslashes} unnecessary line continuations")
else:
    print("❌ Syntax errors still found:")
    print(result.stderr)
    # エラーがあった場合はバックアップを復元
    print("\nRestoring backup...")
    file_path.write_text(content, encoding='utf-8')
    print("Original content restored")
