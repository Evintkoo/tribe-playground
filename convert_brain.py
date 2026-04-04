"""
Convert BrainVISA/MNI polygon format to standard Wavefront OBJ.
Format layout:
  Line 1:  P ambient diffuse specular shininess opacity num_vertices
  Lines 2..1+N: vertex coords (x y z per line)
  empty line
  N normal coords (x y z per line)
  empty line
  polygon_count
  surface_descriptor line
  empty line
  polygon_count cumulative end-indices  (all multiples of 3 → all triangles)
  polygon_count*3 vertex indices (0-based)
"""
import sys

INPUT  = "brain-surface.obj"
OUTPUT = "brain.obj"

print("Reading", INPUT, "...")

with open(INPUT, "r") as f:
    # Header
    header = f.readline().split()
    num_verts = int(header[6])
    print(f"  {num_verts} vertices")

    # Vertices
    verts = []
    for _ in range(num_verts):
        x, y, z = f.readline().split()
        verts.append((float(x), float(y), float(z)))

    f.readline()  # empty line

    # Normals
    normals = []
    for _ in range(num_verts):
        x, y, z = f.readline().split()
        normals.append((float(x), float(y), float(z)))

    f.readline()  # empty line

    # Polygon count
    poly_count = int(f.readline().strip())
    print(f"  {poly_count} polygons")

    f.readline()  # surface descriptor line (e.g. " 0 1 1 1 1")
    f.readline()  # empty line

    # Cumulative end-indices (one per polygon, all multiples of 3 → triangles)
    end_indices = []
    while len(end_indices) < poly_count:
        line = f.readline()
        if line.strip():
            end_indices.extend(int(x) for x in line.split())
    total_idx = end_indices[-1]  # should be poly_count * 3
    print(f"  {total_idx} index entries  (triangles: {total_idx // 3})")

    # Vertex indices (0-based)
    indices = []
    while len(indices) < total_idx:
        line = f.readline()
        if line.strip():
            indices.extend(int(x) for x in line.split())

print("Writing", OUTPUT, "...")

with open(OUTPUT, "w") as f:
    f.write("# Brain surface mesh converted from MNI/BrainVISA format\n")
    f.write(f"# {num_verts} vertices, {poly_count} triangles\n\n")

    for x, y, z in verts:
        f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

    f.write("\n")
    for x, y, z in normals:
        f.write(f"vn {x:.6f} {y:.6f} {z:.6f}\n")

    f.write("\ng brain\n")
    for i in range(poly_count):
        start = end_indices[i - 1] if i > 0 else 0
        a = indices[start]     + 1  # OBJ is 1-indexed
        b = indices[start + 1] + 1
        c = indices[start + 2] + 1
        f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")

import os
size_mb = os.path.getsize(OUTPUT) / 1_048_576
print(f"Done. {OUTPUT}  {size_mb:.1f} MB")
