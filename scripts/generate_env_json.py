import os
import json
import re

def main():
    # Resolve workspace root dynamically
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(root_dir, "local.env")
    if not os.path.exists(env_path):
        env_path = os.path.join(root_dir, ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(root_dir, ".env.example")
        
    print(f"Reading environment variables from: {env_path}")
    
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(r"^\s*([\w.-]+)\s*=\s*(.*)\s*$", line)
                if match:
                    key = match.group(1)
                    value = match.group(2).strip()
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    env_vars[key] = value
                
    # Define exact target output locations
    targets = [
        os.path.join(root_dir, "apps", "mobile", "env.json"),
        os.path.join(root_dir, "apps", "web", "src", "env.json")
    ]
    
    for target_path in targets:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(env_vars, f, indent=2)
        print(f"Generated environment JSON at: {target_path}")

if __name__ == "__main__":
    main()
