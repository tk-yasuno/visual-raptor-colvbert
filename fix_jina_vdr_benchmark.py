#!/usr/bin/env python3
"""
Fix jina_vdr_benchmark.py by replacing escaped newlines
"""

import pathlib
import subprocess

# ファイルを読み込む
file_path = pathlib.Path('jina_vdr_benchmark.py')
print(f"Reading {file_path}...")

content = file_path.read_text(encoding='utf-8')
print(f"Original file size: {len(content)} characters")

# エスケープされた\nの数を数える
escaped_count = content.count('\\n')
print(f"Found {escaped_count} escaped newlines (\\n)")

if escaped_count == 0:
    print("No escaped newlines found. File is OK.")
    exit(0)

# エスケープされた\nを実際の改行に置換
fixed_content = content.replace('\\n', '\n')

print(f"Fixed file size: {len(fixed_content)} characters")

# バックアップを作成
backup_path = file_path.with_suffix('.py.backup2')
file_path.rename(backup_path)
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
else:
    print("❌ Syntax errors found:")
    print(result.stderr)
    # エラーがあった場合はバックアップを復元
    print("\nRestoring backup...")
    file_path.unlink()
    backup_path.rename(file_path)
    print("Backup restored due to errors")
