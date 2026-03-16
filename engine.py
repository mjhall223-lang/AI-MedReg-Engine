import io
import pypdf
from pathlib import Path

def list_all_laws():
    """
    ULTRA-RECURSIVE: Hunts for PDFs specifically for your 
    Regulations/Regulations structure.
    """
    laws = []
    try:
        # Start at the root Regulations folder
        path_root = Path("Regulations")
        if path_root.exists():
            # .rglob('*') finds PDFs in ANY subfolder (Colorado, Federal, etc.)
            laws = sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')])
    except Exception:
        pass # Prevents the blank screen if pathing is locked
    return laws
