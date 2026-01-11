"""
Entity Threading for Character-Based Narrative Learning
========================================================
Extracts character-specific paragraphs from novels to create "Entity Threads"
that force the model to learn long-term character arcs and relationships.

This addresses a key limitation of chunk-based learning: breaking narrative
continuity. By extracting all paragraphs mentioning a character and 
concatenating them, we create a continuous "thread" of that character's story.

Example:
    For "The Count of Monte Cristo", extract all paragraphs mentioning Dantès,
    creating a continuous narrative of his journey from sailor → prisoner → count.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional


# Default character lists for the novels in the dataset
# Use tuples to explicitly group character name variants
DEFAULT_CHARACTERS = {
    "The Count of Monte Cristo": [
        ("Dantès", ["Dantès", "Edmond", "Monte Cristo"]),  # Protagonist
        ("Villefort", ["Villefort"]),
        ("Fernand", ["Fernand", "Mondego"]),
        ("Danglars", ["Danglars"]),
        ("Mercédès", ["Mercédès", "Mercedes"]),
        ("Faria", ["Faria", "Abbé"]),
        ("Morrel", ["Maximilian", "Morrel"]),
        ("Haydée", ["Haydée"]),
        ("Valentine", ["Valentine"]),
        ("Albert", ["Albert"]),
        ("Caderousse", ["Caderousse"]),
        ("Noirtier", ["Noirtier"]),
        ("Bertuccio", ["Bertuccio"])
    ],
    "In search of the castaways": [
        ("Paganel", ["Paganel", "Jacques"]),
        ("Glenarvan", ["Glenarvan", "Edward"]),
        ("Thalcave", ["Thalcave"]),
        ("Mary Grant", ["Mary"]),
        ("Robert Grant", ["Robert"]),
        ("Captain Grant", ["Captain Grant", "Harry Grant"]),
        ("McNabbs", ["McNabbs", "Major", "Mac-Nabbs"]),
        ("Ayrton", ["Ayrton", "Ben Joyce"]),
        ("Lady Glenarvan", ["Lady Helena", "Helena"]),
        ("Mulready", ["Mulready"]),
        ("Wilson", ["Wilson"]),
        ("Olbinett", ["Olbinett"])
    ]
}


def _clean_gutenberg(text: str) -> str:
    """Remove Project Gutenberg headers and footers."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG"
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG"
    ]
    
    start_idx = 0
    for marker in start_markers:
        if marker in text:
            start_idx = text.find(marker)
            start_idx = text.find('\n', start_idx) + 1
            break
    
    end_idx = len(text)
    for marker in end_markers:
        if marker in text:
            end_idx = text.find(marker)
            break
    
    return text[start_idx:end_idx].strip()


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs (separated by blank lines)."""
    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Clean up whitespace and filter empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Filter out very short paragraphs (likely headers/chapter titles)
    paragraphs = [p for p in paragraphs if len(p.split()) >= 5]
    
    return paragraphs


def extract_character_paragraphs(
    paragraphs: List[str],
    character_names: List[str]
) -> List[str]:
    """
    Extract paragraphs that mention any of the character's names.
    
    Args:
        paragraphs: List of text paragraphs
        character_names: List of name variations for the character
        
    Returns:
        List of paragraphs mentioning the character
    """
    matching_paragraphs = []
    
    # Create regex pattern for case-insensitive matching
    # Use word boundaries to avoid partial matches
    pattern = r'\b(' + '|'.join(re.escape(name) for name in character_names) + r')\b'
    regex = re.compile(pattern, re.IGNORECASE)
    
    for paragraph in paragraphs:
        if regex.search(paragraph):
            matching_paragraphs.append(paragraph)
    
    return matching_paragraphs


def create_character_threads(
    novel_path: Path,
    output_dir: Path,
    char_list: Optional[List[str]] = None,
    min_paragraphs: int = 10
) -> List[Path]:
    """
    Create character-specific thread files for a novel.
    
    This function extracts all paragraphs mentioning specific characters
    and concatenates them into separate "thread" files. These threads
    capture the full arc of each character's story.
    
    Args:
        novel_path: Path to the novel text file
        output_dir: Directory to save thread files
        char_list: Optional list of character names to extract.
                   If None, uses default character list for the novel.
        min_paragraphs: Minimum number of paragraphs required to create a thread
        
    Returns:
        List of paths to created thread files
    """
    novel_name = novel_path.stem
    print(f"\n📖 Creating character threads for: {novel_name}")
    
    # Load and clean novel text
    with open(novel_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text = _clean_gutenberg(text)
    paragraphs = _split_into_paragraphs(text)
    print(f"   📝 Found {len(paragraphs):,} paragraphs")
    
    # Get character list
    if char_list is None:
        # Try to find matching default characters
        for key, chars in DEFAULT_CHARACTERS.items():
            if key.lower() in novel_name.lower():
                char_list = chars
                break
        
        if char_list is None:
            print(f"   ⚠️ No default characters for '{novel_name}', extracting top entities...")
            char_list = _extract_top_entities(text, top_k=10)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process character list (check if it's tuples or old format)
    if char_list and isinstance(char_list[0], tuple):
        # New format: list of (primary_name, [variants])
        character_groups = {primary: variants for primary, variants in char_list}
    else:
        # Old format: flat list - use old grouping logic
        character_groups = _group_character_names(char_list)
    
    thread_paths = []
    
    for primary_name, name_variants in character_groups.items():
        # Extract paragraphs for this character
        matching = extract_character_paragraphs(paragraphs, name_variants)
        
        if len(matching) >= min_paragraphs:
            # Create thread file
            thread_filename = f"thread_{primary_name.lower().replace(' ', '_')}.txt"
            thread_path = output_dir / thread_filename
            
            # Concatenate paragraphs with double newlines
            thread_content = "\n\n".join(matching)
            
            with open(thread_path, 'w', encoding='utf-8') as f:
                f.write(thread_content)
            
            thread_paths.append(thread_path)
            print(f"   ✓ Created thread for {primary_name}: {len(matching)} paragraphs")
        else:
            print(f"   ⊘ Skipped {primary_name}: only {len(matching)} paragraphs (min: {min_paragraphs})")
    
    print(f"   ✅ Created {len(thread_paths)} character threads")
    
    return thread_paths


def _group_character_names(char_list: List[str]) -> Dict[str, List[str]]:
    """
    Group character names by primary name.
    Consecutive items with the primary first are considered variants.
    
    Example: ["Dantès", "Edmond", "Monte Cristo"] -> {"Dantès": ["Dantès", "Edmond", "Monte Cristo"]}
    """
    grouped = {}
    current_primary = None
    
    for name in char_list:
        # Check if this is a variant (short name or title)
        is_variant = (
            current_primary is not None and
            len(name.split()) == 1 and
            name not in grouped
        )
        
        if is_variant:
            grouped[current_primary].append(name)
        else:
            current_primary = name
            grouped[name] = [name]
    
    return grouped


def _extract_top_entities(text: str, top_k: int = 10) -> List[str]:
    """
    Extract the most frequently mentioned capitalized words (likely character names).
    Used as fallback when no default character list is available.
    """
    words = text.split()
    capitalized = {}
    
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        if len(clean) > 2 and clean[0].isupper() and not clean.isupper():
            capitalized[clean] = capitalized.get(clean, 0) + 1
    
    # Sort by frequency and return top entities
    sorted_entities = sorted(capitalized.items(), key=lambda x: x[1], reverse=True)
    
    return [name for name, count in sorted_entities[:top_k] if count > 50]


def create_all_threads(
    novels_dir: Path,
    output_dir: Path
) -> Dict[str, List[Path]]:
    """
    Create character threads for all novels in a directory.
    
    Args:
        novels_dir: Directory containing novel .txt files
        output_dir: Base directory for thread output
        
    Returns:
        Dictionary mapping novel names to their thread file paths
    """
    novels_dir = Path(novels_dir)
    output_dir = Path(output_dir)
    
    all_threads = {}
    
    for novel_path in novels_dir.glob("*.txt"):
        # Create novel-specific output subdirectory
        novel_output = output_dir / novel_path.stem
        
        threads = create_character_threads(novel_path, novel_output)
        all_threads[novel_path.stem] = threads
    
    return all_threads


# ==============================================================================
# Testing
# ==============================================================================

def test_entity_threading():
    """Test entity threading on sample novel."""
    print("=" * 60)
    print("TESTING ENTITY THREADING")
    print("=" * 60)
    
    ROOT = Path(__file__).resolve().parents[2]
    novels_dir = ROOT / "Dataset" / "Books"
    output_dir = ROOT / "Dataset" / "entity_threads"
    
    # Test on The Count of Monte Cristo
    novel_path = novels_dir / "The Count of Monte Cristo.txt"
    
    if novel_path.exists():
        threads = create_character_threads(
            novel_path=novel_path,
            output_dir=output_dir / "monte_cristo",
            min_paragraphs=10
        )
        
        print(f"\n📚 Created {len(threads)} thread files:")
        for path in threads:
            size_kb = path.stat().st_size / 1024
            print(f"   - {path.name}: {size_kb:.1f} KB")
    else:
        print(f"⚠️ Novel not found: {novel_path}")
    
    print("\n✅ Entity threading test complete")


if __name__ == "__main__":
    test_entity_threading()
