"""Export the official Popo tree without inventing page-level Markdown."""
from __future__ import annotations

from collections import Counter
import html
from pathlib import Path
import shutil

from common import write_json, write_text


def walk(node):
    yield node
    # Official supplementary leaf nodes have no children field.
    for child in node.get("children", []):
        yield from walk(child)


def export_tree(tree, blocks, asset_root: Path, output: Path):
    nodes = list(walk(tree))
    expected = Counter(block["id"] for block in blocks)
    actual = Counter(i for node in nodes for i in node["block_ids"])
    if any(count != 1 for count in expected.values()) or actual != expected:
        raise ValueError(f"Popo tree block coverage mismatch: missing={expected - actual}, extra={actual - expected}")
    by_id = {block["id"]: block for block in blocks}
    chunks, entries = [], []
    asset_root = asset_root.resolve()
    for node in nodes:
        kind, content = node["type"], node["content"]
        if not isinstance(content, str):
            raise ValueError("Tree content must be a string")
        if kind == "root":
            continue
        if kind == "text":
            # Distinguish an actual title from the builder's synthetic Default Title.
            titles = [by_id[i] for i in node["block_ids"]
                      if by_id[i]["type"] == "title" and by_id[i]["level"] >= 0]
            if titles:
                level, title = node["level"], node["title"]
                if level < 1:
                    raise ValueError(f"Cannot render title level {level}")
                chunks.append("#" * level + " " + title if level <= 6 else
                              f'<div role="heading" aria-level="{level}">{html.escape(title)}</div>')
        if kind in {"image", "table", "chart", "seal", "image_block"}:
            asset = node["img_path"]
            if kind == "image" and not asset:
                raise ValueError("Image node has no img_path")
            if asset:
                source = (asset_root / asset).resolve()
                relative = source.relative_to(asset_root)  # reject absolute/traversal escapes
                if not source.is_file():
                    raise FileNotFoundError(source)
                destination = output / "images" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                image_path = destination.relative_to(output).as_posix()
                # Tables use their HTML; retain crops for provenance, not duplicate rendering.
                if kind != "table":
                    chunks.append(f"![](<{image_path}>)")
                entries.append({"image_path": image_path, "source_img_path": asset,
                                "block_ids": node["block_ids"], "locations": node["location"]})
            for value in (node["title"],):
                if value:
                    chunks.append(value)
        if content:
            chunks.append(content.replace("<|txt_split|>", "\n\n").replace("<|txt_contd|>", ""))
        if kind in {"image", "table", "chart", "seal", "image_block"}:
            if node["caption"] and node["caption"] != node["title"]:
                chunks.append(node["caption"])
            if node["metadata"]:
                chunks.append(node["metadata"])
    write_json(output / "document_tree.json", tree)
    write_json(output / "popo_blocks.json", blocks)
    write_text(output / "document.md", "\n\n".join(chunks) + "\n")
    write_json(output / "image_boxes.json", {
        "method": "mineru_344_2605_popo", "entry_count": len(entries), "entries": entries,
        "coordinate_format": "xyxy_01", "page_numbering": "one_based",
        "notes": "Locations and relations are preserved from the official document tree; no page Markdown.",
    })
    return {"input_blocks": len(blocks), "tree_blocks": sum(actual.values()), "assets": len(entries)}
