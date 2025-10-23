#!/usr/bin/env python3
"""
Fix jina_vdr_benchmark.py - complete rewrite approach
"""

import pathlib

file_path = pathlib.Path('jina_vdr_benchmark.py')
print(f"Fixing {file_path}...")

# 元のファイルを読み込む
content = file_path.read_text(encoding='utf-8')
print(f"Original size: {len(content)} characters")

# エスケープされた\nを実際の改行に置換
fixed_content = content.replace('\\n', '\n')
print(f"Fixed size: {len(fixed_content)} characters")

# バックアップ
backup_path = file_path.with_name('jina_vdr_benchmark_original.py')
if not backup_path.exists():
    file_path.rename(backup_path)
    print(f"Backup: {backup_path}")
else:
    file_path.unlink()
    print("Removed old file")

# 修正版を書き込む
file_path.write_text(fixed_content, encoding='utf-8')
print(f"Written: {file_path}")

# 構文チェック
import subprocess
result = subprocess.run(['python', '-m', 'py_compile', str(file_path)], 
                       capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Syntax check PASSED!")
else:
    print("❌ Syntax errors:")
    print(result.stderr[:500])
    # 復元
    file_path.unlink()
    backup_path.rename(file_path)
    print("Restored original")