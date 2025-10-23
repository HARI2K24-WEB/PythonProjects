import py_compile, pathlib
root = pathlib.Path(r'D:/WEB/pyhton')
failed = []
for p in root.rglob('*.py'):
    try:
        py_compile.compile(str(p), doraise=True)
    except Exception as e:
        print('COMPILE FAIL:', p, e)
        failed.append((p,e))
print('Done. failures:', len(failed))
