import os
import requests

# --- CONSTANTS ---
BASE_DIR = os.path.join(os.path.dirname(__file__), "src", "queries")

ZED_BASE = "https://raw.githubusercontent.com/zed-industries/zed/main/crates/languages/src"
TS_BASE = "https://raw.githubusercontent.com/tree-sitter"
NVIM_BASE = "https://raw.githubusercontent.com/nvim-treesitter/nvim-treesitter/master/queries"

# --- THE MASTER MAP ---
# Explicitly defines the best source for EVERY language in your extension map.
URL_MAP = {
    # Web / JS
    "javascript": f"{ZED_BASE}/javascript/outline.scm",
    "typescript": f"{ZED_BASE}/typescript/outline.scm",
    "tsx":        f"{ZED_BASE}/tsx/outline.scm",
    "vue":        f"{ZED_BASE}/vue/outline.scm",
    "css":        f"{ZED_BASE}/css/outline.scm",
    "json":       f"{ZED_BASE}/json/outline.scm",
    "html":       f"{NVIM_BASE}/html/highlights.scm", # HTML has no standard 'tags', using highlights as fallback foundation
    
    # Backend / Core
    "python":     f"{ZED_BASE}/python/outline.scm",
    "go":         f"{ZED_BASE}/go/outline.scm",
    "rust":       f"{ZED_BASE}/rust/outline.scm",
    "java":       f"{TS_BASE}/tree-sitter-java/master/queries/tags.scm",
    "ruby":       f"{TS_BASE}/tree-sitter-ruby/master/queries/tags.scm",
    "php":        f"{TS_BASE}/tree-sitter-php/master/queries/tags.scm",
    "c_sharp":    f"{TS_BASE}/tree-sitter-c-sharp/master/queries/tags.scm",
    "cpp":        f"{ZED_BASE}/cpp/outline.scm",
    "c":          f"{ZED_BASE}/c/outline.scm",
    
    # Systems / Scripting
    "bash":       f"{TS_BASE}/tree-sitter-bash/master/queries/tags.scm",
    "lua":        f"{ZED_BASE}/lua/outline.scm",
    "yaml":       f"{ZED_BASE}/yaml/outline.scm",
    "toml":       f"{ZED_BASE}/toml/outline.scm",
    
    # Config / Infra
    "hcl":        f"{TS_BASE}/tree-sitter-hcl/master/queries/tags.scm", # Terraform
    "dockerfile": f"{ZED_BASE}/dockerfile/outline.scm",
    "make":       f"{NVIM_BASE}/make/highlights.scm", # Make has no 'tags', using highlights fallback
    
    # Docs
    "markdown":   f"{ZED_BASE}/markdown/outline.scm",
}

# --- MANUAL FALLBACKS ---
# For languages where NO remote 'tags.scm' exists (like HTML/Make), we write a minimal one.
FALLBACK_CONTENT = {
    "html": """
(element
  (start_tag (tag_name) @name)
  (#match? @name "^(h[1-6]|div|section|article|main|header|footer)$")) @definition.section
""",
    "make": """
(rule 
  targets: (word) @name) @definition.target
""",
    "vue": """
(script_element) @definition.script
(style_element) @definition.style
"""
}

def setup_queries():
    print(f"🚀 Setting up queries for {len(URL_MAP)} languages in {BASE_DIR}...")
    
    for lang, url in URL_MAP.items():
        target_dir = os.path.join(BASE_DIR, lang)
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(target_dir, "tags.scm")
        
        print(f"   ⬇️  {lang:<12} : ", end="")
        
        # 1. Try Download
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(target_file, "w") as f:
                    f.write(resp.text)
                print("✅ Downloaded")
                continue
            else:
                print(f"⚠️  HTTP {resp.status_code} (URL: {url})")
        except Exception as e:
            print(f"❌ Network Error: {e}")

        # 2. Apply Fallback (if download failed or known missing)
        fallback = FALLBACK_CONTENT.get(lang)
        if fallback:
            print(f"      ↳ 🔄 Applying manual fallback...", end="")
            with open(target_file, "w") as f:
                f.write(fallback)
            print(" ✅ OK")
        else:
            print(f"      ↳ ❌ No query available. Indexing for {lang} will skip symbols.")

if __name__ == "__main__":
    setup_queries()