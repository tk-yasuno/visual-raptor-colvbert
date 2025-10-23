"""
jina_vdr_benchmark.pyのエスケープされた改行を修正
"""

def fix_escaped_newlines():
    file_path = "jina_vdr_benchmark.py"
    
    # ファイルを読み込み
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # エスケープされた\nを実際の改行に置換
    # ただし、文字列リテラル内の\\nは保持する
    original_content = content
    
    # まずファイル全体をチェック
    count = content.count('\\n')
    print(f"Found {count} instances of escaped newlines")
    
    # エスケープされた\nを実際の改行に置換
    # ただし、三重引用符内や文字列リテラル内のものは注意深く処理
    fixed_content = content.replace('\\n            \\n            ', '\n            \n            ')
    fixed_content = fixed_content.replace('\\n            ', '\n            ')
    fixed_content = fixed_content.replace('\\n                ', '\n                ')
    fixed_content = fixed_content.replace('\\n            )', '\n            )')
    fixed_content = fixed_content.replace('\\n        ', '\n        ')
    fixed_content = fixed_content.replace('\\n    ', '\n    ')
    fixed_content = fixed_content.replace('\\n"""\n', '\n"""\n')
    fixed_content = fixed_content.replace('\\n        }\n', '\n        }\n')
    
    # 変更があった場合のみ書き込み
    if fixed_content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"✅ Fixed escaped newlines in {file_path}")
        
        # 構文チェック
        import py_compile
        try:
            py_compile.compile(file_path, doraise=True)
            print("✅ Syntax check passed!")
        except py_compile.PyCompileError as e:
            print(f"❌ Syntax error found: {e}")
    else:
        print("No changes needed")

if __name__ == "__main__":
    fix_escaped_newlines()
