#!/usr/bin/env python
"""
Fix the active_company bug in app.py
"""
import re

# Read the file
with open('app.py', 'r') as f:
    content = f.read()

# Replace all instances of active_company = config.get_active_company()["id"]
pattern = r'active_company = config\.get_active_company\(\)\["id"\]'
replacement = 'active_company = config.get_active_company()'

# Replace and count replacements
new_content, count = re.subn(pattern, replacement, content)

print(f"Fixed {count} instances of the issue.")

# Write the fixed content back
with open('app.py', 'w') as f:
    f.write(new_content)

print("File updated successfully.")
