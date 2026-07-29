# Sketchfab Downloader

Rust CLI that exports Sketchfab viewer models to self-contained glTF 2.0 (`.glb`) files.

It handles model decryption, protected textures, PBR and shadeless materials, transparency, skeletons, skin weights, morph targets, static poses, and bone/object animation clips.

## Usage

```bash
cargo run --release -- <sketchfab-url-or-uid> [output.glb]
```

Example:

```bash
cargo run --release -- \
  https://sketchfab.com/3d-models/example-0123456789abcdef0123456789abcdef \
  model.glb
```

The first run builds the release binary and downloads the required viewer data. Reusable files are stored in `.cache/`.

## Output

The generated GLB can be imported into Blender or any glTF 2.0-compatible application. Textures, materials, meshes, armatures, skin weights, inverse-bind matrices, and supported animations are embedded in the file.

Use only with models you are authorized to download and reuse.
