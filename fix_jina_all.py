"""Fix all escaped newlines in jina_vdr_benchmark.py"""

def fix_file():
    file_path = "jina_vdr_benchmark.py"
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    fixed_lines = []
    for i, line in enumerate(lines):
        # Skip lines that are inside triple-quoted strings (docstrings or multiline strings)
        # For now, simply replace \\n that appears outside of string literals
        
        # Check if line contains \\n not in a string literal
        if '\\n' in line and not line.strip().startswith(('"""', "'''", '#', 'return """')):
            # This is likely an escaped newline that should be a real newline
            # Replace \\n with actual newline, but be careful
            
            # Special handling for specific patterns
            if line.rstrip().endswith('\\n'):
                # Line ends with \\n, likely should be split
                fixed_line = line.replace('\\n', '\n')
                # Remove any trailing characters after the replaced newline
                parts = fixed_line.split('\n')
                # Keep only the first part and add newline
                fixed_lines.append(parts[0] + '\n')
                # Add remaining parts as new lines
                for part in parts[1:]:
                    if part.strip():
                        fixed_lines.append(' ' * (len(line) - len(line.lstrip())) + part + '\n')
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Fixed {file_path}")
    
    # Verify syntax
    import py_compile
    try:
        py_compile.compile(file_path, doraise=True)
        print("✅ Syntax check passed!")
        return True
    except Exception as e:
        print(f"❌ Syntax error: {e}")
        return False

if __name__ == "__main__":
    fix_file()
