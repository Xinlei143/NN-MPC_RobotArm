from __future__ import annotations

import argparse
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert simple Collada DAE triangle/polylist meshes to ASCII STL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dae", type=Path)
    parser.add_argument("output_stl", type=Path)
    return parser.parse_args()


def _find(root: ET.Element, path: str) -> ET.Element | None:
    return root.find(path, NS)


def _findall(root: ET.Element, path: str) -> list[ET.Element]:
    return root.findall(path, NS)


def _floats(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(item) for item in text.split()]


def _ints(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(item) for item in text.split()]


def _id_ref(value: str) -> str:
    return value[1:] if value.startswith("#") else value


def _matrix_from_node(node: ET.Element) -> np.ndarray:
    matrix_el = _find(node, "c:matrix")
    if matrix_el is None:
        return np.eye(4, dtype=float)
    values = _floats(matrix_el.text)
    if len(values) != 16:
        raise ValueError(f"Collada matrix must contain 16 values, got {len(values)}")
    return np.asarray(values, dtype=float).reshape(4, 4)


def _source_arrays(mesh: ET.Element) -> dict[str, np.ndarray]:
    sources: dict[str, np.ndarray] = {}
    for source in _findall(mesh, "c:source"):
        source_id = source.attrib["id"]
        float_array = _find(source, "c:float_array")
        accessor = _find(source, "c:technique_common/c:accessor")
        if float_array is None or accessor is None:
            continue
        stride = int(accessor.attrib.get("stride", "1"))
        values = np.asarray(_floats(float_array.text), dtype=float)
        if values.size % stride != 0:
            raise ValueError(f"Source {source_id!r} values are not divisible by stride={stride}")
        sources[source_id] = values.reshape((-1, stride))
    return sources


def _vertex_position_source(mesh: ET.Element) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for vertices in _findall(mesh, "c:vertices"):
        vertex_id = vertices.attrib["id"]
        position_input = None
        for input_el in _findall(vertices, "c:input"):
            if input_el.attrib.get("semantic") == "POSITION":
                position_input = input_el
                break
        if position_input is None:
            raise ValueError(f"vertices {vertex_id!r} has no POSITION input")
        mapping[vertex_id] = _id_ref(position_input.attrib["source"])
    return mapping


def _input_stride_and_vertex_offset(primitive: ET.Element) -> tuple[int, int, str]:
    inputs = _findall(primitive, "c:input")
    if not inputs:
        raise ValueError("Collada primitive has no input elements")
    stride = max(int(input_el.attrib.get("offset", "0")) for input_el in inputs) + 1
    for input_el in inputs:
        if input_el.attrib.get("semantic") == "VERTEX":
            return stride, int(input_el.attrib.get("offset", "0")), _id_ref(input_el.attrib["source"])
    raise ValueError("Collada primitive has no VERTEX input")


def _transform_vertex(matrix: np.ndarray, vertex: np.ndarray) -> np.ndarray:
    hom = np.ones(4, dtype=float)
    hom[:3] = vertex[:3]
    return (matrix @ hom)[:3]


def _triangulate_polygon(vertices: list[np.ndarray]) -> list[np.ndarray]:
    if len(vertices) < 3:
        return []
    return [np.asarray([vertices[0], vertices[idx], vertices[idx + 1]], dtype=float) for idx in range(1, len(vertices) - 1)]


def _triangles_from_primitive(
    primitive: ET.Element,
    sources: dict[str, np.ndarray],
    vertex_sources: dict[str, str],
    matrix: np.ndarray,
) -> list[np.ndarray]:
    stride, vertex_offset, vertex_id = _input_stride_and_vertex_offset(primitive)
    position_source = vertex_sources[vertex_id]
    positions = sources[position_source]
    indices = _ints(_find(primitive, "c:p").text if _find(primitive, "c:p") is not None else "")
    if len(indices) % stride != 0:
        raise ValueError("Primitive index list length is not divisible by input stride")
    vertex_indices = indices[vertex_offset::stride]

    tag = primitive.tag.rsplit("}", 1)[-1]
    polygons: list[list[int]]
    if tag == "triangles":
        polygons = [vertex_indices[idx : idx + 3] for idx in range(0, len(vertex_indices), 3)]
    elif tag == "polylist":
        vcounts = _ints(_find(primitive, "c:vcount").text if _find(primitive, "c:vcount") is not None else "")
        polygons = []
        cursor = 0
        for count in vcounts:
            polygons.append(vertex_indices[cursor : cursor + count])
            cursor += count
        if cursor != len(vertex_indices):
            raise ValueError("polylist vcount does not match index list")
    else:
        return []

    triangles: list[np.ndarray] = []
    for polygon in polygons:
        vertices = [_transform_vertex(matrix, positions[index]) for index in polygon]
        triangles.extend(_triangulate_polygon(vertices))
    return triangles


def extract_collada_triangles(path: Path) -> list[np.ndarray]:
    root = ET.parse(path).getroot()
    geometries = {geom.attrib["id"]: geom for geom in _findall(root, "c:library_geometries/c:geometry")}
    triangles: list[np.ndarray] = []
    for node in _findall(root, ".//c:node"):
        matrix = _matrix_from_node(node)
        for instance in _findall(node, "c:instance_geometry"):
            geom_id = _id_ref(instance.attrib["url"])
            geometry = geometries.get(geom_id)
            if geometry is None:
                raise ValueError(f"instance_geometry references unknown geometry {geom_id!r}")
            mesh = _find(geometry, "c:mesh")
            if mesh is None:
                continue
            sources = _source_arrays(mesh)
            vertex_sources = _vertex_position_source(mesh)
            for primitive in [*_findall(mesh, "c:triangles"), *_findall(mesh, "c:polylist")]:
                triangles.extend(_triangles_from_primitive(primitive, sources, vertex_sources, matrix))
    if not triangles:
        raise ValueError(f"No triangles found in {path}")
    return triangles


def _normal(triangle: np.ndarray) -> np.ndarray:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12 or not math.isfinite(norm):
        return np.zeros(3, dtype=float)
    return normal / norm


def write_binary_stl(path: Path, triangles: list[np.ndarray], solid_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(triangles) > 200000:
        raise ValueError(f"STL face count must be <= 200000 for MuJoCo, got {len(triangles)}")
    header = f"converted from Collada: {solid_name}".encode("ascii", errors="ignore")[:80]
    header = header + b" " * (80 - len(header))
    with path.open("wb") as file:
        file.write(header)
        file.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            normal = _normal(triangle).astype(np.float32)
            file.write(struct.pack("<3f", *normal.tolist()))
            for vertex in triangle:
                file.write(struct.pack("<3f", *vertex.astype(np.float32).tolist()))
            file.write(struct.pack("<H", 0))


def main() -> None:
    args = parse_args()
    triangles = extract_collada_triangles(args.input_dae)
    write_binary_stl(args.output_stl, triangles, args.output_stl.stem)
    print(f"converted {args.input_dae} -> {args.output_stl} triangles={len(triangles)}")


if __name__ == "__main__":
    main()
