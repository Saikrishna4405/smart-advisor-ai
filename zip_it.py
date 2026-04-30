import os
import zipfile

exclude_dirs = {'__pycache__', 'venv', 'env', '.venv', '.git', '.idea', '.vscode', 'scratch', 'investment-mobile'}
exclude_exts = {'.zip', '.db', '.log'}

with zipfile.ZipFile('Fresh_Deploy.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if not any(file.endswith(ext) for ext in exclude_exts) and file != '.env':
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, '.')
                if rel_path != 'zip_it.py':
                    zf.write(file_path, rel_path)
print("done")
