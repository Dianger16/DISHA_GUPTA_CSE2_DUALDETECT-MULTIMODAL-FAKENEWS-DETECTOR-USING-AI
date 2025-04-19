import os

print("Index.html Absolute Path:", os.path.abspath("templates/index.html"))
print("Exists:", os.path.exists("templates/index.html"))  # Should return True
