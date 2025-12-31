import os

def normalize_names():
    root = "model-dev/data/train"
    if not os.path.exists(root):
        print(f"Path not found: {root}")
        return

    renames = {
        "diamonds": "diamond",
        "ovals": "oval",
        "squiggles": "squiggle"
    }

    count = 0
    for dirname in os.listdir(root):
        if not os.path.isdir(os.path.join(root, dirname)):
            continue
            
        new_name = dirname
        for plural, singular in renames.items():
            if new_name.endswith(plural):
                new_name = new_name.replace(plural, singular)
        
        if new_name != dirname:
            src = os.path.join(root, dirname)
            dst = os.path.join(root, new_name)
            print(f"Renaming {dirname} -> {new_name}")
            os.rename(src, dst)
            count += 1
            
    print(f"Renamed {count} folders.")

if __name__ == "__main__":
    normalize_names()
