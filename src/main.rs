use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use clap::Parser;
use flate2::read::GzDecoder;
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgba, imageops::FilterType};
use rand::Rng;
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use wasmtime::{
    Caller, Engine as WasmEngine, ExternType, Func, Linker, Memory, MemoryType, Module, Store, Val,
    ValType,
};

const STATIC_KEY: &str = "77d92dd656ac3fdde472d5ba59747f42ac0ce217";
const CACHE_DIR: &str = ".cache";

#[derive(Parser)]
#[command(version, about = "Download Sketchfab embed models as glTF 2.0 GLB")]
struct Args {
    sketchfab_url_or_uid: String,
    output: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct TextureEntry {
    uid: String,
    url: String,
    pk: Option<u32>,
    filename: String,
    clean_file: Option<String>,
}

#[derive(Clone, Debug)]
struct TextureUse {
    uid: String,
    texcoord_unit: u32,
    transform: TextureTransform,
    sampler: SamplerSettings,
    alpha_channel: bool,
}

#[derive(Clone, Debug)]
struct TextureTransform {
    offset: [f32; 2],
    scale: [f32; 2],
    rotation: f32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SamplerSettings {
    mag_filter: u32,
    min_filter: u32,
    wrap_s: u32,
    wrap_t: u32,
}

#[derive(Clone, Debug)]
struct MaterialEntry {
    name: String,
    base_color: [f32; 4],
    base_color_texture: Option<TextureUse>,
    emissive_color: [f32; 3],
    emissive_enabled: bool,
    emissive_texture: Option<TextureUse>,
    occlusion_texture: Option<TextureUse>,
    normal_texture: Option<TextureUse>,
    metallic_texture: Option<TextureUse>,
    roughness_texture: Option<TextureUse>,
    roughness_invert: bool,
    opacity_texture: Option<TextureUse>,
    alpha_mask_texture: Option<TextureUse>,
    alpha_invert: bool,
    normal_scale: f32,
    normal_flip_y: bool,
    metallic_factor: f32,
    roughness_factor: f32,
    alpha_mode: &'static str,
    alpha_cutoff: f32,
    double_sided: bool,
    unlit: bool,
    extensions: Map<String, Value>,
}

#[derive(Clone, Debug)]
struct AnimationEntry {
    uid: String,
    name: String,
    url: String,
}

#[derive(Debug)]
struct ModelConfig {
    work_dir: PathBuf,
    base_url: String,
    diter_b: String,
    static_key: String,
    texture_map: HashMap<String, TextureEntry>,
    materials: HashMap<String, MaterialEntry>,
    animations: Vec<AnimationEntry>,
    vertex_colors: bool,
    vertex_color_alpha: bool,
    vertex_color_srgb: bool,
    flip_uvs: bool,
}

#[derive(Clone)]
enum AttrData {
    F32(Vec<f32>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    U16(Vec<u16>),
    U8(Vec<u8>),
    I16(Vec<i16>),
}

impl AttrData {
    fn as_i64_vec(&self) -> Vec<i64> {
        match self {
            Self::F32(v) => v.iter().map(|x| *x as i64).collect(),
            Self::I32(v) => v.iter().map(|x| *x as i64).collect(),
            Self::U32(v) => v.iter().map(|x| *x as i64).collect(),
            Self::U16(v) => v.iter().map(|x| *x as i64).collect(),
            Self::U8(v) => v.iter().map(|x| *x as i64).collect(),
            Self::I16(v) => v.iter().map(|x| *x as i64).collect(),
        }
    }

    fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Self::F32(v) => v.clone(),
            Self::I32(v) => v.iter().map(|x| *x as f32).collect(),
            Self::U32(v) => v.iter().map(|x| *x as f32).collect(),
            Self::U16(v) => v.iter().map(|x| *x as f32).collect(),
            Self::U8(v) => v.iter().map(|x| *x as f32).collect(),
            Self::I16(v) => v.iter().map(|x| *x as f32).collect(),
        }
    }
}

#[derive(Clone)]
struct Attribute {
    data: AttrData,
    item_size: usize,
    count: usize,
    component_type: u32,
    normalized: bool,
}

struct Geometry {
    indices: Vec<u32>,
    mode: u32,
    attributes: HashMap<String, Attribute>,
    morph_targets: Vec<MorphTarget>,
    texcoord_units: HashMap<u32, u32>,
    material_name: Option<String>,
    joint_names: Vec<String>,
    matrix: [f32; 16],
    skeleton_matrix: Option<[f32; 16]>,
    animation_target: Option<String>,
}

struct MorphTarget {
    name: String,
    attributes: HashMap<String, Attribute>,
}

#[derive(Clone, Copy)]
struct MorphTargetBinding {
    node: usize,
    target_index: usize,
    target_count: usize,
}

#[derive(Clone)]
struct ScalarTrack {
    times: Vec<f32>,
    values: Vec<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let uid_re = Regex::new(r"([a-f0-9]{32})")?;
    let uid = uid_re
        .captures(&args.sketchfab_url_or_uid)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_owned())
        .ok_or_else(|| anyhow!("could not extract 32-character model UID"))?;
    let output = args
        .output
        .unwrap_or_else(|| PathBuf::from(format!("{uid}.glb")));

    println!("Sketchfab Downloader - Model: {uid}\n");
    let config = get_model_config(&uid)?;
    println!("  Base URL: {}", config.base_url);
    let mut texture_names = config.texture_map.keys().cloned().collect::<Vec<_>>();
    texture_names.sort();
    println!(
        "  Textures: {}\n",
        if texture_names.is_empty() {
            "none".to_string()
        } else {
            texture_names.join(", ")
        }
    );

    download_files(&config)?;
    decrypt_all(&config)?;
    let texture_files = match descramble_textures(&config) {
        Ok(map) => map,
        Err(err) => {
            eprintln!("  Texture descramble failed: {err}");
            config.texture_map.clone()
        }
    };

    let osgjs: Value = serde_json::from_slice(&fs::read(config.work_dir.join("file.osgjs"))?)?;
    let poly_bin = fs::read(config.work_dir.join("model_file.bin"))?;
    let wire_path = config.work_dir.join("model_file_wireframe.bin");
    let wire_bin = if wire_path.exists() {
        Some(fs::read(wire_path)?)
    } else {
        None
    };
    let animation_bins = config
        .animations
        .iter()
        .map(|animation| {
            Ok((
                animation.name.to_ascii_lowercase(),
                fs::read(
                    config
                        .work_dir
                        .join("animations")
                        .join(format!("{}.bin", animation.uid)),
                )?,
            ))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let glb = convert_to_glb(
        &osgjs,
        &poly_bin,
        wire_bin.as_deref(),
        &texture_files,
        &config.materials,
        &animation_bins,
        &config.work_dir,
        config.vertex_colors,
        config.vertex_color_alpha,
        config.vertex_color_srgb,
        config.flip_uvs,
    )?;
    fs::write(&output, &glb)?;
    println!(
        "\n[6/6] Done! {} ({:.1} MB)",
        output.display(),
        glb.len() as f64 / 1024.0 / 1024.0
    );
    Ok(())
}

fn fetch(url: &str) -> Result<Vec<u8>> {
    let response = reqwest::blocking::get(url).with_context(|| format!("fetch {url}"))?;
    if !response.status().is_success() {
        bail!("fetch {url} failed with {}", response.status());
    }
    Ok(response.bytes()?.to_vec())
}

fn fetch_text(url: &str) -> Result<String> {
    Ok(String::from_utf8(fetch(url)?)?)
}

fn ensure_wasm(embed_html: &str) -> Result<()> {
    let wasm_path = Path::new("deobfuscated").join("decrypt.wasm");
    if wasm_path.exists() {
        return Ok(());
    }

    println!("  Extracting decrypt.wasm from viewer bundles...");
    let bundle_re =
        Regex::new(r#"https://static\.sketchfab\.com/static/builds/web/dist/[^\"&]+\.js"#)?;
    let mut bundle_urls = bundle_re
        .find_iter(embed_html)
        .map(|m| m.as_str().to_owned())
        .collect::<Vec<_>>();
    bundle_urls.sort();
    bundle_urls.dedup();

    if bundle_urls.is_empty() {
        bail!("could not find Sketchfab viewer JS bundles in embed page");
    }

    for url in bundle_urls {
        let js = fetch_text(&url)?;
        let Some(wasm_idx) = js.find("AGFzbQ") else {
            continue;
        };
        let start = js[..wasm_idx]
            .rfind('"')
            .map(|i| i + 1)
            .ok_or_else(|| anyhow!("malformed viewer bundle containing WASM"))?;
        let mut end = wasm_idx;
        while end < js.len() {
            if js.as_bytes()[end] == b'"' && (end == 0 || js.as_bytes()[end - 1] != b'\\') {
                break;
            }
            end += 1;
        }
        if end >= js.len() {
            continue;
        }

        let encoded = js[start..end].replace("\\n", "");
        let wasm_bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .with_context(|| format!("decode embedded WASM from {url}"))?;
        if wasm_bytes.starts_with(b"\0asm") {
            fs::create_dir_all("deobfuscated")?;
            fs::write(&wasm_path, &wasm_bytes)?;
            println!(
                "  decrypt.wasm: {} bytes (from {})",
                wasm_bytes.len(),
                url.rsplit('/').next().unwrap_or("viewer bundle")
            );
            return Ok(());
        }
    }

    bail!("could not find embedded WASM decryption module in viewer bundles")
}

fn extract_static_key(embed_html: &str) -> Result<String> {
    let bundle_re =
        Regex::new(r#"https://static\.sketchfab\.com/static/builds/web/dist/[^\"&]+\.js"#)?;
    let mut bundle_urls = bundle_re
        .find_iter(embed_html)
        .map(|m| m.as_str().to_owned())
        .collect::<Vec<_>>();
    bundle_urls.sort();
    bundle_urls.dedup();

    let key_re_1 = Regex::new(
        r#"exports\s*\.\s*k\s*:\s*\(\)\s*=>\s*\w+\}\s*;\s*const\s+\w+\s*=\s*"([0-9a-f]{40})\\n"#,
    )?;
    let key_re_2 =
        Regex::new(r#"\{k:\s*\(\)\s*=>\s*\w+\}[^;]*;\s*const\s+\w+\s*=\s*"([0-9a-f]{40})"#)?;
    // Current builds expose the key through the C04p module.
    let key_re_3 = Regex::new(
        r#"C04p\s*:\s*\w+\s*=>\s*\{\s*"use strict";\s*\w+\.exports\s*=\s*"([0-9a-f]{40})\\n"#,
    )?;
    for url in bundle_urls {
        let js = fetch_text(&url)?;
        if let Some(caps) = key_re_1.captures(&js) {
            return Ok(caps[1].to_ascii_lowercase());
        }
        if let Some(caps) = key_re_2.captures(&js) {
            return Ok(caps[1].to_ascii_lowercase());
        }
        if let Some(caps) = key_re_3.captures(&js) {
            return Ok(caps[1].to_ascii_lowercase());
        }
    }

    Ok(STATIC_KEY.to_owned())
}

fn prefetched_data(embed_html: &str) -> Result<Value> {
    let marker = r#"id="js-dom-data-prefetched-data"><!--"#;
    let start = embed_html
        .find(marker)
        .map(|i| i + marker.len())
        .ok_or_else(|| anyhow!("prefetched model data not found"))?;
    let end = embed_html[start..]
        .find("--></div>")
        .map(|i| start + i)
        .ok_or_else(|| anyhow!("unterminated prefetched model data"))?;
    let json_text = embed_html[start..end]
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'");
    Ok(serde_json::from_str(&json_text)?)
}

fn enabled_channel<'a>(material: &'a Value, name: &str) -> Option<&'a Value> {
    let channel = material.pointer(&format!("/channels/{name}"))?;
    channel
        .get("enable")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        .then_some(channel)
}

fn texture_filter(value: Option<&str>, default: u32) -> u32 {
    match value {
        Some("NEAREST") => 9728,
        Some("LINEAR") => 9729,
        Some("NEAREST_MIPMAP_NEAREST") => 9984,
        Some("LINEAR_MIPMAP_NEAREST") => 9985,
        Some("NEAREST_MIPMAP_LINEAR") => 9986,
        Some("LINEAR_MIPMAP_LINEAR") => 9987,
        _ => default,
    }
}

fn texture_wrap(value: Option<&str>) -> u32 {
    match value {
        Some("CLAMP_TO_EDGE") => 33071,
        Some("MIRRORED_REPEAT") => 33648,
        _ => 10497,
    }
}

fn vec2(value: Option<&Value>, default: [f32; 2]) -> [f32; 2] {
    let Some(values) = value.and_then(Value::as_array) else {
        return default;
    };
    [
        values
            .first()
            .and_then(Value::as_f64)
            .unwrap_or(default[0] as f64) as f32,
        values
            .get(1)
            .and_then(Value::as_f64)
            .unwrap_or(default[1] as f64) as f32,
    ]
}

fn texture_use(channel: Option<&Value>) -> Option<TextureUse> {
    let channel = channel?;
    let texture = channel.get("texture")?;
    Some(TextureUse {
        uid: texture.get("uid")?.as_str()?.to_owned(),
        texcoord_unit: texture
            .get("texCoordUnit")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        transform: TextureTransform {
            offset: vec2(channel.pointer("/UVTransforms/offset"), [0.0, 0.0]),
            scale: vec2(channel.pointer("/UVTransforms/scale"), [1.0, 1.0]),
            rotation: channel
                .pointer("/UVTransforms/rotation")
                .and_then(Value::as_f64)
                .unwrap_or(0.0) as f32,
        },
        sampler: SamplerSettings {
            mag_filter: texture_filter(texture.get("magFilter").and_then(Value::as_str), 9729),
            min_filter: texture_filter(texture.get("minFilter").and_then(Value::as_str), 9987),
            wrap_s: texture_wrap(texture.get("wrapS").and_then(Value::as_str)),
            wrap_t: texture_wrap(texture.get("wrapT").and_then(Value::as_str)),
        },
        alpha_channel: texture.get("internalFormat").and_then(Value::as_str) == Some("ALPHA"),
    })
}

fn color3(channel: Option<&Value>, default: [f32; 3]) -> [f32; 3] {
    let Some(values) = channel
        .and_then(|v| v.get("color"))
        .and_then(Value::as_array)
    else {
        return default;
    };
    let mut out = default;
    for (i, value) in values.iter().take(3).enumerate() {
        out[i] = value.as_f64().unwrap_or(default[i] as f64) as f32;
    }
    out
}

fn get_model_config(uid: &str) -> Result<ModelConfig> {
    println!("[1/6] Fetching embed page...");
    let html = fetch_text(&format!("https://sketchfab.com/models/{uid}/embed"))?
        .replace("&#34;", "\"")
        .replace("&quot;", "\"");
    ensure_wasm(&html)?;
    let static_key = extract_static_key(&html)?;

    let p_re = Regex::new(r#""p"\s*:\s*\[\{[^}]*"v"\s*:\s*(\d+)[^}]*"b"\s*:\s*"([^"]+)""#)?;
    let binz_re =
        Regex::new(r#"https://media\.sketchfab\.com/models/[^"]*/files/[^"]*/file\.binz"#)?;
    let p_caps = p_re
        .captures(&html)
        .ok_or_else(|| anyhow!("could not extract diter config"))?;
    let binz = binz_re
        .find(&html)
        .ok_or_else(|| anyhow!("could not extract file.binz URL"))?
        .as_str();
    let base_url = binz.trim_end_matches("/file.binz").to_owned();

    let prefetched = prefetched_data(&html)?;
    let model_key = format!("/i/models/{uid}");
    let textures_key = format!("/i/models/{uid}/textures?optimized=1");
    let animations_key = format!("/i/models/{uid}/animations?optimized=1");
    let model = prefetched
        .get(&model_key)
        .ok_or_else(|| anyhow!("model metadata not found in prefetched data"))?;

    let mut materials = HashMap::new();
    let mut wanted_textures = HashSet::new();
    let global_unlit = model
        .pointer("/options/shading/type")
        .and_then(Value::as_str)
        == Some("shadeless");
    if let Some(source_materials) = model
        .pointer("/options/materials")
        .and_then(Value::as_object)
    {
        for material in source_materials.values().filter(|v| v.is_object()) {
            let Some(name) = material.get("name").and_then(Value::as_str) else {
                continue;
            };
            let diffuse = enabled_channel(material, "DiffusePBR")
                .or_else(|| enabled_channel(material, "AlbedoPBR"));
            let emissive = enabled_channel(material, "EmitColor");
            let occlusion = enabled_channel(material, "AOPBR");
            let normal = enabled_channel(material, "NormalMap");
            let metallic = enabled_channel(material, "MetalnessPBR");
            let roughness_channel = enabled_channel(material, "RoughnessPBR");
            let glossiness_channel = enabled_channel(material, "GlossinessPBR");
            let roughness = roughness_channel.or(glossiness_channel);
            let opacity = enabled_channel(material, "Opacity");
            let alpha_mask = enabled_channel(material, "AlphaMask");
            let base_color_texture = texture_use(diffuse);
            let emissive_texture = texture_use(emissive);
            let occlusion_texture = texture_use(occlusion);
            let normal_texture = texture_use(normal);
            let metallic_texture = texture_use(metallic);
            let roughness_texture = texture_use(roughness);
            let opacity_texture = texture_use(opacity);
            let alpha_mask_texture = texture_use(alpha_mask);
            for texture in [
                base_color_texture.as_ref(),
                emissive_texture.as_ref(),
                occlusion_texture.as_ref(),
                normal_texture.as_ref(),
                metallic_texture.as_ref(),
                roughness_texture.as_ref(),
                opacity_texture.as_ref(),
                alpha_mask_texture.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                wanted_textures.insert(texture.uid.clone());
            }
            let base_rgb = color3(diffuse, [1.0, 1.0, 1.0]);
            let emissive_factor = emissive
                .and_then(|v| v.get("factor"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0) as f32;
            let mut emissive_color = color3(emissive, [1.0, 1.0, 1.0]);
            for value in &mut emissive_color {
                *value *= emissive_factor;
            }
            let alpha_mode = if alpha_mask.is_some() {
                "MASK"
            } else if opacity.is_some() {
                "BLEND"
            } else {
                "OPAQUE"
            };
            let alpha = opacity
                .and_then(|v| v.get("factor"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0) as f32;
            let mut extensions = Map::new();
            if let Some(clearcoat) = enabled_channel(material, "ClearCoat") {
                let factor = clearcoat
                    .get("factor")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0) as f32;
                let roughness = enabled_channel(material, "ClearCoatRoughness")
                    .and_then(|channel| channel.get("factor"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0) as f32;
                extensions.insert(
                    "KHR_materials_clearcoat".to_owned(),
                    json!({
                        "clearcoatFactor": factor,
                        "clearcoatRoughnessFactor": roughness
                    }),
                );
            }
            if let Some(specular) = enabled_channel(material, "SpecularF0") {
                extensions.insert(
                    "KHR_materials_specular".to_owned(),
                    json!({
                        "specularFactor": specular
                            .get("factor")
                            .and_then(Value::as_f64)
                            .unwrap_or(1.0)
                    }),
                );
            }
            materials.insert(
                name.to_owned(),
                MaterialEntry {
                    name: name.to_owned(),
                    base_color: [base_rgb[0], base_rgb[1], base_rgb[2], alpha],
                    base_color_texture,
                    emissive_color,
                    emissive_enabled: emissive.is_some(),
                    emissive_texture,
                    occlusion_texture,
                    normal_texture,
                    metallic_texture,
                    roughness_texture,
                    roughness_invert: glossiness_channel.is_some() && roughness_channel.is_none(),
                    opacity_texture,
                    alpha_mask_texture,
                    alpha_invert: opacity
                        .or(alpha_mask)
                        .and_then(|channel| channel.get("invert"))
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    normal_scale: normal
                        .and_then(|channel| channel.get("factor"))
                        .and_then(Value::as_f64)
                        .unwrap_or(1.0) as f32,
                    normal_flip_y: normal
                        .and_then(|channel| channel.get("flipY"))
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    metallic_factor: metallic
                        .and_then(|v| v.get("factor"))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    roughness_factor: roughness
                        .and_then(|v| v.get("factor"))
                        .and_then(Value::as_f64)
                        .map(|value| {
                            if glossiness_channel.is_some() && roughness_channel.is_none() {
                                1.0 - value
                            } else {
                                value
                            }
                        })
                        .unwrap_or(1.0) as f32,
                    alpha_mode,
                    alpha_cutoff: alpha_mask
                        .and_then(|v| v.get("factor"))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.5) as f32,
                    double_sided: material
                        .get("cullFace")
                        .and_then(Value::as_str)
                        .is_some_and(|value| value == "DISABLE"),
                    unlit: global_unlit
                        || material
                            .get("shadeless")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                    extensions,
                },
            );
        }
    }

    let mut texture_map = HashMap::new();
    if let Some(texture_sets) = prefetched
        .get(&textures_key)
        .and_then(|v| v.get("results"))
        .and_then(Value::as_array)
    {
        for texture_set in texture_sets {
            let Some(set_uid) = texture_set.get("uid").and_then(Value::as_str) else {
                continue;
            };
            if !wanted_textures.contains(set_uid) {
                continue;
            }
            let best = texture_set
                .get("images")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|image| image.get("url").and_then(Value::as_str).is_some())
                .max_by_key(|image| image.get("width").and_then(Value::as_u64).unwrap_or(0));
            let Some(image) = best else {
                continue;
            };
            let url = image["url"].as_str().unwrap().to_owned();
            let source_name = url.rsplit('/').next().unwrap_or("texture.png");
            let extension = Path::new(source_name)
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("png")
                .to_owned();
            texture_map.insert(
                set_uid.to_owned(),
                TextureEntry {
                    uid: set_uid.to_owned(),
                    url,
                    pk: image.get("pk").and_then(Value::as_u64).map(|v| v as u32),
                    filename: format!("{set_uid}.{extension}"),
                    clean_file: None,
                },
            );
        }
    }

    let animations = prefetched
        .get(&animations_key)
        .and_then(|v| v.get("results"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|animation| {
            Some(AnimationEntry {
                uid: animation.get("uid")?.as_str()?.to_owned(),
                name: animation.get("name")?.as_str()?.to_owned(),
                url: animation.get("url")?.as_str()?.to_owned(),
            })
        })
        .collect();
    let vertex_color = model.pointer("/options/shading/vertexColor");

    Ok(ModelConfig {
        work_dir: Path::new(CACHE_DIR).join(uid),
        base_url,
        diter_b: p_caps[2].to_owned(),
        static_key,
        texture_map,
        materials,
        animations,
        vertex_colors: vertex_color
            .and_then(|value| value.get("enable"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        vertex_color_alpha: vertex_color
            .and_then(|value| value.get("useAlpha"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        vertex_color_srgb: vertex_color
            .and_then(|value| value.get("colorSpace"))
            .and_then(Value::as_str)
            == Some("srgb"),
        flip_uvs: !global_unlit,
    })
}

fn download_files(config: &ModelConfig) -> Result<()> {
    println!("[2/6] Downloading model files...");
    fs::create_dir_all(config.work_dir.join("textures"))?;
    fs::create_dir_all(config.work_dir.join("animations"))?;
    for name in ["file.binz", "model_file.binz", "model_file_wireframe.binz"] {
        let dest = config.work_dir.join(name);
        if !dest.exists() {
            let data = fetch(&format!("{}/{}", config.base_url, name))?;
            fs::write(&dest, &data)?;
            println!("  {name}: {} bytes", data.len());
        }
    }
    for (channel, tex) in &config.texture_map {
        let dest = config.work_dir.join("textures").join(&tex.filename);
        if !dest.exists() {
            let data = fetch(&tex.url)?;
            fs::write(&dest, &data)?;
            println!("  {channel} texture: {} bytes", data.len());
        }
    }
    for animation in &config.animations {
        let compressed = config
            .work_dir
            .join("animations")
            .join(format!("{}.bin.gz", animation.uid));
        let raw = config
            .work_dir
            .join("animations")
            .join(format!("{}.bin", animation.uid));
        if !compressed.exists() {
            let data = fetch(&animation.url)?;
            fs::write(&compressed, &data)?;
            println!("  {} animation: {} bytes", animation.name, data.len());
        }
        if !raw.exists() {
            let mut decoder = GzDecoder::new(fs::File::open(&compressed)?);
            let mut data = Vec::new();
            decoder.read_to_end(&mut data)?;
            fs::write(&raw, &data)?;
        }
    }
    Ok(())
}

fn parse_wasm_data_size(bytes: &[u8]) -> u32 {
    let mut m = 65536u32;
    let mut d = 8usize;
    while d < bytes.len() {
        let y = read_leb(bytes, &mut d);
        let len = read_leb(bytes, &mut d) as usize;
        let h = d.saturating_add(len);
        if y > 11 || len == 0 || h > bytes.len() {
            break;
        }
        if y == 6 {
            let _ = read_leb(bytes, &mut d);
            d += 2;
            let _ = read_leb(bytes, &mut d);
            m = read_leb(bytes, &mut d);
        }
        if y == 11 {
            let count = read_leb(bytes, &mut d);
            for _ in 0..count {
                d += 1;
                let _ = read_leb(bytes, &mut d);
                let _ = read_leb(bytes, &mut d);
                let _ = read_leb(bytes, &mut d);
                let size = read_leb(bytes, &mut d) as usize;
                d += size;
                if d >= h {
                    break;
                }
            }
        }
        d = h;
    }
    m
}

fn read_leb(bytes: &[u8], offset: &mut usize) -> u32 {
    let start = *offset;
    let mut n = 0u32;
    let mut e = 0x80u8;
    while e & 0x80 != 0 && *offset < bytes.len() {
        e = bytes[*offset];
        n |= ((e & 0x7f) as u32) << (7 * (*offset - start));
        *offset += 1;
    }
    n
}

struct WasmState {
    current_break: u32,
}

fn decrypt_binz(path: &Path, diter_b: &str, static_key: &str) -> Result<Vec<u8>> {
    let wasm_path = Path::new("deobfuscated").join("decrypt.wasm");
    if !wasm_path.exists() {
        bail!("decrypt.wasm not found at deobfuscated/decrypt.wasm");
    }
    let wasm_bytes = fs::read(&wasm_path)?;
    let encrypted = fs::read(path)?;
    let data_size = parse_wasm_data_size(&wasm_bytes);
    let initial_pages = (262144 + (((data_size + 65535) >> 16) << 16)) >> 16;

    let engine = WasmEngine::default();
    let module = Module::from_binary(&engine, &wasm_bytes)?;
    let mut store = Store::new(
        &engine,
        WasmState {
            current_break: data_size,
        },
    );
    let memory = Memory::new(
        &mut store,
        MemoryType::new(initial_pages, Some(536870912 >> 16)),
    )?;
    let mut linker = Linker::new(&engine);
    linker.define(&mut store, "env", "memory", memory)?;
    let sbrk_memory = memory;
    linker.func_wrap(
        "env",
        "sbrk",
        move |mut caller: Caller<'_, WasmState>, inc: i32| -> i32 {
            let old = caller.data().current_break;
            let new_break = old.wrapping_add(inc as u32);
            let overflow = new_break as i64 - sbrk_memory.data_size(&caller) as i64;
            if overflow > 0 {
                let _ = sbrk_memory.grow(&mut caller, (overflow as u64 + 65535) >> 16);
            }
            caller.data_mut().current_break = new_break;
            old as i32
        },
    )?;
    let time_memory = memory;
    linker.func_wrap(
        "env",
        "time",
        move |mut caller: Caller<'_, WasmState>, t: i32| -> i32 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i32;
            if t != 0 {
                let _ = time_memory.write(&mut caller, t as usize, &now.to_le_bytes());
            }
            now
        },
    )?;
    let gettimeofday_memory = memory;
    linker.func_wrap(
        "env",
        "gettimeofday",
        move |mut caller: Caller<'_, WasmState>, t: i32| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            if t != 0 {
                let secs = now.as_secs() as u32;
                let usecs = now.subsec_micros();
                let _ = gettimeofday_memory.write(&mut caller, t as usize, &secs.to_le_bytes());
                let _ =
                    gettimeofday_memory.write(&mut caller, t as usize + 4, &usecs.to_le_bytes());
            }
        },
    )?;
    linker.func_wrap("env", "abort", || -> Result<()> { bail!("WASM abort") })?;
    for import in module.imports() {
        if import.module() != "env"
            || matches!(
                import.name(),
                "memory" | "sbrk" | "time" | "gettimeofday" | "abort"
            )
        {
            continue;
        }
        let ExternType::Func(func_ty) = import.ty() else {
            continue;
        };
        let result_types = func_ty.results().collect::<Vec<_>>();
        let func = Func::new(
            &mut store,
            func_ty.clone(),
            move |_caller, _params, results| {
                for (result, ty) in results.iter_mut().zip(result_types.iter()) {
                    *result = zero_val(ty);
                }
                Ok(())
            },
        );
        linker.define(&mut store, "env", import.name(), func)?;
    }

    let instance = linker.instantiate(&mut store, &module)?;
    if let Some(f) = instance
        .get_typed_func::<(), ()>(&mut store, "__wasm_call_ctors")
        .ok()
    {
        f.call(&mut store, ())?;
    }
    let alloc_input =
        instance.get_typed_func::<i32, i32>(&mut store, "heSBnb29kYnllCk5ldmVyIGdvbm5hIHRl")?;
    let reset =
        instance.get_typed_func::<(), ()>(&mut store, "mV2ZXIgZ29ubmEgbGV0IHlvdSBkb3duCk5l")?;
    let rick_rolled = instance.get_typed_func::<(i32, i32), i32>(&mut store, "Umlja1JvbGxlZDRV")?;
    let alloc_diter_b =
        instance.get_typed_func::<i32, i32>(&mut store, "dmVyIGdvbm5hIHJ1biBhcm91bmQgYW5kI")?;
    // The WASM export takes no parameters.
    let process =
        instance.get_typed_func::<(), i32>(&mut store, "GRlc2VydCB5b3UKTmV2ZXIgZ29ubmEgbW")?;
    let advance =
        instance.get_typed_func::<(), ()>(&mut store, "FrZSB5b3UgY3J5Ck5ldmVyIGdvbm5hIHN")?;
    let get_info =
        instance.get_typed_func::<(), i32>(&mut store, "bGwgYSBsaWUgYW5kIGh1cnQgeW91Cg")?;
    let get_start =
        instance.get_typed_func::<(), i32>(&mut store, "TmV2ZXIgZ29ubmEgZ2l2ZSB5b3UgdXAKT")?;

    let seed = 1314 + rand::thread_rng().gen_range(0..9999i32);
    let key_hex = static_key[..40].to_ascii_lowercase();
    let mut collected = Vec::new();
    let mut running = seed as u32;
    for i in 0..10 {
        let g = u32::from_str_radix(&key_hex[4 * i..4 * i + 4], 16)?;
        running ^= g;
        collected.push(g ^ seed as u32);
        collected.push(running);
    }
    let mut xor_all = collected[19];
    for t in 0..10 {
        xor_all ^= collected[2 * t];
    }
    let key_off = rick_rolled.call(&mut store, (seed, 40))? as usize;
    for t in 0..10 {
        let hex = format!("{:04x}", collected[2 * t] ^ xor_all);
        memory.write(&mut store, key_off + 4 * t, hex.as_bytes())?;
    }

    let diter_clean = diter_b.replace("\\n", "").replace('\n', "");
    let diter_bytes = base64::engine::general_purpose::STANDARD.decode(diter_clean)?;
    reset.call(&mut store, ())?;
    let diter_off = alloc_diter_b.call(&mut store, diter_bytes.len() as i32)? as usize;
    memory.write(&mut store, diter_off, &diter_bytes)?;
    process.call(&mut store, ())?;

    let mut chunks = Vec::new();
    for chunk in encrypted.chunks(10240) {
        let input_off = alloc_input.call(&mut store, chunk.len() as i32)? as usize;
        memory.write(&mut store, input_off, chunk)?;
        let mut more = process.call(&mut store, ())?;
        while more != 0 {
            let start = get_start.call(&mut store, ())? as usize;
            let len = get_info.call(&mut store, ())? as usize;
            let mut out = vec![0u8; len];
            memory.read(&store, start, &mut out)?;
            chunks.extend(out);
            advance.call(&mut store, ())?;
            more = process.call(&mut store, ())?;
        }
    }
    if chunks.starts_with(&[0x1f, 0x8b]) {
        let mut decoder = GzDecoder::new(chunks.as_slice());
        let mut out = Vec::new();
        decoder.read_to_end(&mut out)?;
        Ok(out)
    } else {
        Ok(chunks)
    }
}

fn zero_val(ty: &ValType) -> Val {
    match ty {
        ValType::I32 => Val::I32(0),
        ValType::I64 => Val::I64(0),
        ValType::F32 => Val::F32(0),
        ValType::F64 => Val::F64(0),
        ValType::V128 => Val::V128(0.into()),
        ValType::Ref(r) => Val::null_ref(r.heap_type()),
    }
}

fn decrypt_all(config: &ModelConfig) -> Result<()> {
    println!("[3/6] Decrypting model files...");
    for (src, dst) in [
        ("file.binz", "file.osgjs"),
        ("model_file.binz", "model_file.bin"),
        ("model_file_wireframe.binz", "model_file_wireframe.bin"),
    ] {
        let dst_path = config.work_dir.join(dst);
        if dst_path.exists() {
            continue;
        }
        let result = decrypt_binz(
            &config.work_dir.join(src),
            &config.diter_b,
            &config.static_key,
        )?;
        fs::write(&dst_path, &result)?;
        println!("  {dst}: {} bytes", result.len());
    }
    Ok(())
}

fn mod_i(i: i64, u: i64) -> i64 {
    i - (i / u) * u
}

fn tri_sum(y: i64, t: i64, f: i64) -> i64 {
    let x = y.min(t);
    let n = y.max(t);
    if f < x {
        return f * (f + 1) / 2;
    }
    if f < n {
        return x * (x + 1) / 2 + x * (f - x);
    }
    let r = f - n;
    x * (x + 1) / 2 + x * (n - x) + (x - 1) * r - (r - 1) * r / 2
}

fn xy_to_zigzag(gw: i64, gh: i64, px: i64, py: i64) -> i64 {
    let r = gw.min(gh);
    let n = gw.max(gh);
    let v = px + py;
    let even = mod_i(v, 2) == 0;
    if v < r {
        return tri_sum(gw, gh, v) + if even { v - py } else { py };
    }
    if v < n {
        let mut s = gh - py - 1;
        if gw < gh {
            s = r - (gw - px);
        }
        return tri_sum(gw, gh, v) + if even { s } else { r - s - 1 };
    }
    let s = gh - py - 1;
    let e = r + n - v - 1;
    tri_sum(gw, gh, v) + if even { s } else { e - s - 1 }
}

fn pixel_to_block_idx(x: i64, y: i64, bw: i64, bh: i64) -> usize {
    let bi = xy_to_zigzag(bw, bh, x / 8, y / 8);
    let rot = mod_i(bi, 4);
    let mut px = mod_i(x, 8);
    let mut py = mod_i(y, 8);
    if rot == 1 {
        px = 7 - px;
    } else if rot == 2 {
        std::mem::swap(&mut px, &mut py);
    } else if rot == 3 {
        let old_px = px;
        px = 7 - py;
        py = old_px;
    }
    (bi * 64 + px + py * 8) as usize
}

fn descramble_textures(config: &ModelConfig) -> Result<HashMap<String, TextureEntry>> {
    println!("[4/6] Descrambling textures...");
    let mut clean = HashMap::new();
    for (uid, tex) in &config.texture_map {
        let src = config.work_dir.join("textures").join(&tex.filename);
        let Some(pk) = tex.pk else {
            clean.insert(uid.clone(), tex.clone());
            continue;
        };
        let clean_name = format!("{}_clean.png", tex.uid);
        let dst = config.work_dir.join("textures").join(&clean_name);
        let mut next = tex.clone();
        next.clean_file = Some(clean_name.clone());
        if dst.exists() {
            clean.insert(uid.clone(), next);
            continue;
        }
        let img = image::open(&src)?.to_rgba8();
        let (w, h) = img.dimensions();
        if w % 8 != 0 || h % 8 != 0 {
            println!("  {uid}: keeping original texture ({w}x{h} is not divisible by 8)");
            clean.insert(uid.clone(), tex.clone());
            continue;
        }
        println!("  {uid}: {w}x{h} pk={pk}");
        let total = (w * h) as usize;
        let offset = ((-(pk as i64) * 64).rem_euclid(total as i64)) as usize;
        let bw = (w / 8) as i64;
        let bh = (h / 8) as i64;
        let mut block_map = vec![0usize; total];
        let mut inv = vec![(0u32, 0u32); total];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let fi = pixel_to_block_idx(x as i64, y as i64, bw, bh);
                block_map[idx] = fi;
                inv[fi] = (x, y);
            }
        }
        let mut out: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let fi = block_map[(y * w + x) as usize];
                let shifted = (fi + offset) % total;
                let (sx, sy) = inv[shifted];
                out.put_pixel(x, y, *img.get_pixel(sx, sy));
            }
        }
        out.save(&dst)?;
        println!("    -> {clean_name}");
        clean.insert(uid.clone(), next);
    }
    Ok(clean)
}

fn num(v: Option<&Value>) -> f64 {
    match v {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::String(s)) => s.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn first_key_value(obj: &Value) -> Option<(&str, &Value)> {
    obj.as_object()?.iter().next().map(|(k, v)| (k.as_str(), v))
}

fn read_buffer_array(
    bin: &[u8],
    def: &Value,
    item_size: usize,
    type_name: &str,
) -> Result<AttrData> {
    let offset = def.get("Offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let size = def
        .get("Size")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("missing buffer Size"))? as usize;
    let count = size * item_size;
    if def.get("Encoding").and_then(Value::as_str) == Some("varint") {
        let signed = !type_name.starts_with('U');
        let vals = decode_varint(&bin[offset..], count, signed)?;
        return if signed {
            Ok(AttrData::I32(vals.into_iter().map(|x| x as i32).collect()))
        } else {
            Ok(AttrData::U32(vals.into_iter().map(|x| x as u32).collect()))
        };
    }
    let bytes = &bin[offset..];
    Ok(match type_name {
        "Float32Array" => AttrData::F32(read_f32(bytes, count)?),
        "Int32Array" => AttrData::I32(read_i32(bytes, count)?),
        "Uint32Array" => AttrData::U32(read_u32(bytes, count)?),
        "Uint16Array" => AttrData::U16(read_u16(bytes, count)?),
        "Uint8Array" => AttrData::U8(bytes[..count].to_vec()),
        "Int16Array" => AttrData::I16(read_i16(bytes, count)?),
        _ => bail!("unknown typed array {type_name}"),
    })
}

fn read_f32(bytes: &[u8], count: usize) -> Result<Vec<f32>> {
    Ok(bytes
        .chunks_exact(4)
        .take(count)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn read_i32(bytes: &[u8], count: usize) -> Result<Vec<i32>> {
    Ok(bytes
        .chunks_exact(4)
        .take(count)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn read_u32(bytes: &[u8], count: usize) -> Result<Vec<u32>> {
    Ok(bytes
        .chunks_exact(4)
        .take(count)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn read_u16(bytes: &[u8], count: usize) -> Result<Vec<u16>> {
    Ok(bytes
        .chunks_exact(2)
        .take(count)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn read_i16(bytes: &[u8], count: usize) -> Result<Vec<i16>> {
    Ok(bytes
        .chunks_exact(2)
        .take(count)
        .map(|b| i16::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn decode_varint(bytes: &[u8], count: usize, signed: bool) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(count);
    let mut off = 0usize;
    while out.len() < count {
        let mut s = 0i64;
        let mut shift = 0;
        loop {
            let b = *bytes
                .get(off)
                .ok_or_else(|| anyhow!("varint buffer ended early"))?;
            off += 1;
            s |= ((b & 127) as i64) << shift;
            shift += 7;
            if b & 128 == 0 {
                break;
            }
        }
        out.push(if signed { (s >> 1) ^ -(s & 1) } else { s });
    }
    Ok(out)
}

fn typed_u32(value: i64, bits: u32) -> u32 {
    if bits == 16 {
        value as u16 as u32
    } else {
        value as u32
    }
}

fn delta_decode(arr: &mut [u32], start: usize, bits: u32) {
    if arr.is_empty() || start >= arr.len() {
        return;
    }
    let mut prev = arr[start];
    for v in arr.iter_mut().skip(start + 1) {
        let x = *v as i64;
        *v = typed_u32(prev as i64 + ((x >> 1) ^ -(x & 1)), bits);
        prev = *v;
    }
}

fn implicit_decode(enc: &[u32], out_len: usize, start_idx: usize, use_expected: bool) -> Vec<u32> {
    let mut out = vec![0u32; out_len];
    let mut r = enc.get(2).copied().unwrap_or(0);
    let mask_len = enc.get(1).copied().unwrap_or(0) as usize;
    let mut idx = start_idx;
    let pad = mask_len * 32 - out_len;
    for u in 0..mask_len {
        let c = enc.get(3 + u).copied().unwrap_or(0);
        let mut h = u * 32;
        for d in if u == mask_len - 1 { pad } else { 0 }..32 {
            if h >= out_len {
                break;
            }
            if c & (0x8000_0000u32 >> d) != 0 {
                out[h] = enc.get(idx).copied().unwrap_or(0);
                idx += 1;
            } else if use_expected {
                out[h] = r;
            } else {
                out[h] = r;
                r += 1;
            }
            h += 1;
        }
    }
    out
}

fn expected_renumber(arr: &mut [u32], state: &mut i64, bits: u32) {
    let mut n = *state;
    for a in arr {
        let o = n - *a as i64;
        *a = typed_u32(o, bits);
        if n <= o {
            n = o + 1;
        }
    }
    *state = n;
}

fn strip_to_tris(indices: &[u32]) -> Vec<u32> {
    let mut tris = Vec::new();
    for i in 0..indices.len().saturating_sub(2) {
        let (a, b, c) = (indices[i], indices[i + 1], indices[i + 2]);
        if a == b || b == c || a == c {
            continue;
        }
        if i % 2 == 0 {
            tris.extend([a, b, c]);
        } else {
            tris.extend([b, a, c]);
        }
    }
    tris
}

fn parallelogram_predict(data: &mut [f32], item_size: usize, strip: &[u32]) {
    if strip.len() < 3 {
        return;
    }
    let mut visited = vec![false; data.len() / item_size];
    for &i in strip.iter().take(3) {
        if let Some(v) = visited.get_mut(i as usize) {
            *v = true;
        }
    }
    for i in 2..strip.len().saturating_sub(1) {
        let (a, b, c, d) = (
            strip[i - 2] as usize,
            strip[i - 1] as usize,
            strip[i] as usize,
            strip[i + 1] as usize,
        );
        if d >= visited.len() || visited[d] {
            continue;
        }
        visited[d] = true;
        for j in 0..item_size {
            data[d * item_size + j] +=
                data[b * item_size + j] + data[c * item_size + j] - data[a * item_size + j];
        }
    }
}

fn dequantize(encoded: &[f32], bbl: &[f32], h: &[f32], item_size: usize) -> Vec<f32> {
    let mut out = vec![0.0; encoded.len()];
    for i in 0..encoded.len() / item_size {
        for j in 0..item_size {
            out[i * item_size + j] = bbl[j] + encoded[i * item_size + j] * h[j];
        }
    }
    out
}

fn decode_normals(encoded: &[f32], item_size: usize, epsilon: f32, nphi: f32) -> Vec<f32> {
    let count = encoded.len() / 2;
    let mut out = vec![0.0; count * item_size];
    let cos_eps = (0.01745329251 * epsilon).cos();
    let d_phi = std::f32::consts::PI / (nphi - 1.0);
    let d_gamma = 1.57079632679 / (nphi - 1.0);
    for i in 0..count {
        let oi = i * item_size;
        let ii = i * 2;
        let mut s = encoded[ii] as i32;
        let x = encoded[ii + 1];
        if item_size == 4 {
            out[oi + 3] = if s & 1024 != 0 { -1.0 } else { 1.0 };
            s &= !1024;
        }
        let a0 = s as f32 * d_phi;
        let r = a0.cos();
        let w = a0.sin();
        let a1 = a0 + d_gamma;
        let e = ((cos_eps - r * a1.cos()) / (1e-5f32.max(w * a1.sin()))).clamp(-1.0, 1.0);
        let p = 6.28318530718 * x / (std::f32::consts::PI / 1e-5f32.max(e.acos())).ceil();
        out[oi] = w * p.cos();
        out[oi + 1] = w * p.sin();
        out[oi + 2] = r;
    }
    out
}

fn meta_value(meta: &HashMap<String, Value>, key: &str) -> f32 {
    num(meta.get(key)) as f32
}

fn morph_delta_attribute(
    target: Vec<f32>,
    base: &Attribute,
    item_size: usize,
    count: usize,
) -> Option<Attribute> {
    if base.item_size != item_size || base.count != count {
        return None;
    }
    let base = base.data.to_f32_vec();
    if target.len() != base.len() {
        return None;
    }
    Some(Attribute {
        data: AttrData::F32(
            target
                .into_iter()
                .zip(base)
                .map(|(target, base)| target - base)
                .collect(),
        ),
        item_size,
        count,
        component_type: 5126,
        normalized: false,
    })
}

fn decode_morph_target(
    target: &Value,
    poly_bin: &[u8],
    base_attributes: &HashMap<String, Attribute>,
) -> Result<Option<MorphTarget>> {
    let name = target
        .get("Name")
        .and_then(Value::as_str)
        .unwrap_or("morph")
        .to_owned();
    let metadata = user_data_values(target);
    let attributes = target.get("VertexAttributeList").and_then(Value::as_object);
    let Some(attributes) = attributes else {
        return Ok(None);
    };
    let mut output = HashMap::new();

    if let Some(definition) = attributes.get("Vertex") {
        if let Some((array_type, array_definition)) =
            definition.get("Array").and_then(first_key_value)
        {
            let item_size = definition
                .get("ItemSize")
                .and_then(Value::as_u64)
                .unwrap_or(3) as usize;
            let count = array_definition
                .get("Size")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let mut positions =
                read_buffer_array(poly_bin, array_definition, item_size, array_type)?.to_f32_vec();
            if metadata.contains_key("vtx_bbl_x") {
                let mut lower = vec![
                    meta_value(&metadata, "vtx_bbl_x"),
                    meta_value(&metadata, "vtx_bbl_y"),
                ];
                let mut step = vec![
                    meta_value(&metadata, "vtx_h_x"),
                    meta_value(&metadata, "vtx_h_y"),
                ];
                if item_size == 3 {
                    lower.push(meta_value(&metadata, "vtx_bbl_z"));
                    step.push(meta_value(&metadata, "vtx_h_z"));
                }
                positions = dequantize(&positions, &lower, &step, item_size);
            }
            if let Some(base) = base_attributes.get("POSITION") {
                if let Some(attribute) = morph_delta_attribute(positions, base, item_size, count) {
                    output.insert("POSITION".to_owned(), attribute);
                }
            }
        }
    }

    if let Some(definition) = attributes.get("Normal") {
        if let Some((array_type, array_definition)) =
            definition.get("Array").and_then(first_key_value)
        {
            let count = array_definition
                .get("Size")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let raw = read_buffer_array(poly_bin, array_definition, 2, array_type)?;
            let normals = decode_normals(
                &raw.to_f32_vec(),
                3,
                meta_value(&metadata, "epsilon").max(0.25),
                meta_value(&metadata, "nphi").max(720.0),
            );
            if let Some(base) = base_attributes.get("NORMAL") {
                if let Some(attribute) = morph_delta_attribute(normals, base, 3, count) {
                    output.insert("NORMAL".to_owned(), attribute);
                }
            }
        }
    }

    Ok((!output.is_empty()).then_some(MorphTarget {
        name,
        attributes: output,
    }))
}

fn process_geometry(
    geom: &Value,
    poly_bin: &[u8],
    wire_bin: Option<&[u8]>,
) -> Result<Option<Geometry>> {
    let material_name = geom
        .pointer("/StateSet/osg.StateSet/AttributeList")
        .and_then(Value::as_array)
        .and_then(|attributes| {
            attributes.iter().find_map(|attribute| {
                attribute
                    .pointer("/osg.Material/Name")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
        })
        .or_else(|| {
            geom.pointer("/StateSet/osg.StateSet/Name")
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
    let mut meta = HashMap::new();
    if let Some(values) = geom
        .pointer("/UserDataContainer/Values")
        .and_then(Value::as_array)
    {
        for v in values {
            if let Some(name) = v.get("Name").and_then(Value::as_str) {
                let raw = v.get("Value").cloned().unwrap_or(Value::Null);
                let parsed = raw
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(Value::from)
                    .unwrap_or(raw);
                meta.insert(name.to_owned(), parsed);
            }
        }
    }

    let mut strip_indices = None;
    let mut indices = Vec::new();
    let mut primitive_mode = None;
    let mut expected_state = 0i64;
    if let Some(prims) = geom.get("PrimitiveSetList").and_then(Value::as_array) {
        for prim in prims {
            let Some((_, draw)) = first_key_value(prim) else {
                continue;
            };
            let Some(arr_info) = draw.pointer("/Indices/Array") else {
                continue;
            };
            let Some((arr_type, arr_def)) = first_key_value(arr_info) else {
                continue;
            };
            if arr_def
                .get("File")
                .and_then(Value::as_str)
                .is_some_and(|file| file.contains("wireframe"))
            {
                continue;
            }
            let bin = if arr_def
                .get("File")
                .and_then(Value::as_str)
                .is_some_and(|s| s.contains("wireframe"))
            {
                if let Some(w) = wire_bin {
                    w
                } else {
                    continue;
                }
            } else {
                poly_bin
            };
            let mode = draw.get("Mode").and_then(Value::as_str);
            let is_strip = mode == Some("TRIANGLE_STRIP");
            let output_mode = match mode {
                Some("POINTS") => 0,
                Some("LINES") => 1,
                Some("LINE_LOOP") => 2,
                Some("LINE_STRIP") => 3,
                Some("TRIANGLES") | Some("TRIANGLE_STRIP") => 4,
                _ => continue,
            };
            if primitive_mode.is_some_and(|current| current != output_mode) {
                continue;
            }
            primitive_mode = Some(output_mode);
            let source_bits = if arr_type == "Uint32Array" { 32 } else { 16 };
            let mut idx: Vec<u32> = read_buffer_array(bin, arr_def, 1, arr_type)?
                .as_i64_vec()
                .into_iter()
                .map(|v| v as u32)
                .collect();
            let tri_mode = num(meta.get("triangle_mode")) as u32;
            let output_bits;
            if tri_mode & 4 != 0 && is_strip {
                let start = 3 + idx.get(1).copied().unwrap_or(0) as usize;
                if tri_mode & 1 != 0 {
                    delta_decode(&mut idx, start, source_bits);
                }
                idx = implicit_decode(
                    &idx,
                    idx.first().copied().unwrap_or(0) as usize,
                    start,
                    tri_mode & 2 != 0,
                );
                for value in &mut idx {
                    *value &= 0xffff;
                }
                output_bits = 16;
            } else if tri_mode & 1 != 0 {
                delta_decode(&mut idx, 0, source_bits);
                output_bits = source_bits;
            } else {
                output_bits = source_bits;
            }
            if tri_mode & 2 != 0 {
                expected_renumber(&mut idx, &mut expected_state, output_bits);
            }
            if is_strip {
                strip_indices = Some(idx.clone());
                indices.extend(strip_to_tris(&idx));
            } else {
                indices.extend(idx);
            }
        }
    }
    if indices.is_empty() {
        return Ok(None);
    }

    let mut attributes = HashMap::new();
    if let Some(va) = geom.get("VertexAttributeList").and_then(Value::as_object) {
        for (name, def) in va {
            let Some(arr_info) = def.get("Array") else {
                continue;
            };
            let Some((arr_type, arr_def)) = first_key_value(arr_info) else {
                continue;
            };
            let bin = if arr_def
                .get("File")
                .and_then(Value::as_str)
                .is_some_and(|s| s.contains("wireframe"))
            {
                if let Some(w) = wire_bin {
                    w
                } else {
                    continue;
                }
            } else {
                poly_bin
            };
            let item_size = def.get("ItemSize").and_then(Value::as_u64).unwrap_or(1) as usize;
            let count = arr_def.get("Size").and_then(Value::as_u64).unwrap_or(0) as usize;
            let raw = read_buffer_array(bin, arr_def, item_size, arr_type)?;
            let attr_flags = num(meta.get("attributes")) as u32;
            if name == "Vertex" {
                let mut data = raw.to_f32_vec();
                if num(meta.get("vertex_mode")) as u32 & 2 != 0 {
                    if let Some(strip) = &strip_indices {
                        parallelogram_predict(&mut data, item_size, strip);
                    }
                }
                if meta.contains_key("vtx_bbl_x") {
                    let mut bbl = vec![
                        meta_value(&meta, "vtx_bbl_x"),
                        meta_value(&meta, "vtx_bbl_y"),
                    ];
                    let mut h = vec![meta_value(&meta, "vtx_h_x"), meta_value(&meta, "vtx_h_y")];
                    if item_size == 3 {
                        bbl.push(meta_value(&meta, "vtx_bbl_z"));
                        h.push(meta_value(&meta, "vtx_h_z"));
                    }
                    data = dequantize(&data, &bbl, &h, item_size);
                }
                attributes.insert(
                    "POSITION".to_owned(),
                    Attribute {
                        data: AttrData::F32(data),
                        item_size,
                        count,
                        component_type: 5126,
                        normalized: false,
                    },
                );
            } else if name == "Normal" && attr_flags & 2 != 0 {
                let data = decode_normals(
                    &raw.to_f32_vec(),
                    3,
                    meta_value(&meta, "epsilon").max(0.25),
                    meta_value(&meta, "nphi").max(720.0),
                );
                attributes.insert(
                    "NORMAL".to_owned(),
                    Attribute {
                        data: AttrData::F32(data),
                        item_size: 3,
                        count,
                        component_type: 5126,
                        normalized: false,
                    },
                );
            } else if name == "Tangent" && attr_flags & 32 != 0 {
                let data = decode_normals(
                    &raw.to_f32_vec(),
                    4,
                    meta_value(&meta, "epsilon").max(0.25),
                    meta_value(&meta, "nphi").max(720.0),
                );
                attributes.insert(
                    "TANGENT".to_owned(),
                    Attribute {
                        data: AttrData::F32(data),
                        item_size: 4,
                        count,
                        component_type: 5126,
                        normalized: false,
                    },
                );
            } else if name.starts_with("TexCoord") {
                let suffix = name.trim_start_matches("TexCoord");
                let mut data = raw.to_f32_vec();
                let uv_mode =
                    meta.get(&format!("uv_{suffix}_mode"))
                        .map(|v| num(Some(v)))
                        .unwrap_or_else(|| num(meta.get("vertex_mode"))) as u32;
                if uv_mode & 2 != 0 {
                    if let Some(strip) = &strip_indices {
                        parallelogram_predict(&mut data, item_size, strip);
                    }
                }
                let prefix = format!("uv_{suffix}_");
                if meta.contains_key(&(prefix.clone() + "bbl_x")) {
                    data = dequantize(
                        &data,
                        &[
                            meta_value(&meta, &(prefix.clone() + "bbl_x")),
                            meta_value(&meta, &(prefix.clone() + "bbl_y")),
                        ],
                        &[
                            meta_value(&meta, &(prefix.clone() + "h_x")),
                            meta_value(&meta, &(prefix + "h_y")),
                        ],
                        item_size,
                    );
                }
                for uv in data.chunks_exact_mut(item_size) {
                    uv[1] = 1.0 - uv[1];
                }
                attributes.insert(
                    format!("_TC_{suffix}"),
                    Attribute {
                        data: AttrData::F32(data),
                        item_size: item_size.max(2),
                        count,
                        component_type: 5126,
                        normalized: false,
                    },
                );
            } else if name == "Color" {
                let normalized = matches!(raw, AttrData::U8(_));
                let component_type = if normalized { 5121 } else { 5126 };
                let data = if normalized {
                    raw
                } else {
                    AttrData::F32(raw.to_f32_vec())
                };
                attributes.insert(
                    "_SKETCHFAB_COLOR_0".to_owned(),
                    Attribute {
                        data,
                        item_size,
                        count,
                        component_type,
                        normalized,
                    },
                );
            }
        }
    }
    let mut tc_keys = attributes
        .keys()
        .filter(|k| k.starts_with("_TC_"))
        .filter_map(|key| {
            key.trim_start_matches("_TC_")
                .parse::<u32>()
                .ok()
                .map(|unit| (unit, key.clone()))
        })
        .collect::<Vec<_>>();
    tc_keys.sort_by_key(|(unit, _)| *unit);
    let mut texcoord_units = HashMap::new();
    for (i, (unit, key)) in tc_keys.into_iter().enumerate() {
        if let Some(attr) = attributes.remove(&key) {
            attributes.insert(format!("TEXCOORD_{i}"), attr);
            texcoord_units.insert(unit, i as u32);
        }
    }
    if !attributes.contains_key("POSITION") {
        return Ok(None);
    }
    let mut morph_targets = Vec::new();
    if let Some(targets) = geom.get("MorphTargets").and_then(Value::as_array) {
        for wrapper in targets {
            let Some((_, target)) = first_key_value(wrapper) else {
                continue;
            };
            match decode_morph_target(target, poly_bin, &attributes) {
                Ok(Some(target)) => morph_targets.push(target),
                Ok(None) => {}
                Err(error) => eprintln!("  Warning: skipping morph target: {error}"),
            }
        }
    }
    Ok(Some(Geometry {
        indices,
        mode: primitive_mode.unwrap_or(4),
        attributes,
        morph_targets,
        texcoord_units,
        material_name,
        joint_names: Vec::new(),
        matrix: matrix16(None),
        skeleton_matrix: None,
        animation_target: None,
    }))
}

fn build_uid_map<'a>(v: &'a Value, map: &mut HashMap<u64, &'a Value>) {
    if let Some(obj) = v.as_object() {
        if obj.len() > 1 {
            if let Some(id) = obj.get("UniqueID").and_then(Value::as_u64) {
                map.insert(id, v);
            }
        }
        for child in obj.values() {
            build_uid_map(child, map);
        }
    } else if let Some(arr) = v.as_array() {
        for child in arr {
            build_uid_map(child, map);
        }
    }
}

fn resolve_refs(v: &mut Value, map: &HashMap<u64, Value>) {
    match v {
        Value::Object(obj) => {
            if obj.len() == 1 {
                if let Some(id) = obj.get("UniqueID").and_then(Value::as_u64) {
                    if let Some(resolved) = map.get(&id) {
                        *v = resolved.clone();
                        return;
                    }
                }
            }
            for child in obj.values_mut() {
                resolve_refs(child, map);
            }
        }
        Value::Array(arr) => {
            for child in arr {
                resolve_refs(child, map);
            }
        }
        _ => {}
    }
}

fn resolved_scene(osgjs: &Value) -> Value {
    let mut uid_refs = HashMap::new();
    build_uid_map(osgjs, &mut uid_refs);
    let owned_map = uid_refs
        .into_iter()
        .map(|(key, value)| (key, value.clone()))
        .collect::<HashMap<_, _>>();
    let mut root = osgjs.clone();
    resolve_refs(&mut root, &owned_map);
    root
}

fn vec_from(value: Option<&Value>, length: usize, fallback: &[f32]) -> Vec<f32> {
    let Some(values) = value.and_then(Value::as_array) else {
        return fallback.to_vec();
    };
    (0..length)
        .map(|index| {
            values
                .get(index)
                .and_then(Value::as_f64)
                .unwrap_or(fallback[index] as f64) as f32
        })
        .collect()
}

fn bone_trs(bone: &Value) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut translation = vec![0.0, 0.0, 0.0];
    let mut rotation = vec![0.0, 0.0, 0.0, 1.0];
    let mut scale = vec![1.0, 1.0, 1.0];
    if let Some(transforms) = bone
        .pointer("/UpdateCallbacks/0/osgAnimation.UpdateBone/StackedTransforms")
        .and_then(Value::as_array)
    {
        for transform in transforms {
            if let Some(value) = transform.pointer("/osgAnimation.StackedTranslate/Translate") {
                translation = vec_from(Some(value), 3, &[0.0, 0.0, 0.0]);
            } else if let Some(value) =
                transform.pointer("/osgAnimation.StackedQuaternion/Quaternion")
            {
                rotation = vec_from(Some(value), 4, &[0.0, 0.0, 0.0, 1.0]);
            } else if let Some(value) = transform.pointer("/osgAnimation.StackedScale/Scale") {
                scale = vec_from(Some(value), 3, &[1.0, 1.0, 1.0]);
            }
        }
    }
    (translation, rotation, scale)
}

fn without_numeric_suffix(name: &str) -> &str {
    name.rsplit_once('_')
        .filter(|(_, suffix)| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
        .map(|(base, _)| base)
        .unwrap_or(name)
}

fn add_bone_node(
    wrapper: &Value,
    nodes: &mut Vec<Value>,
    node_by_name: &mut HashMap<String, usize>,
    inverse_bind_by_name: &mut HashMap<String, [f32; 16]>,
) -> Option<usize> {
    let bone = wrapper.get("osgAnimation.Bone")?;
    let name = bone.get("Name")?.as_str()?.to_owned();
    let (translation, rotation, scale) = bone_trs(bone);
    let index = nodes.len();
    nodes.push(Value::Null);
    let children = bone
        .get("Children")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|child| add_bone_node(child, nodes, node_by_name, inverse_bind_by_name))
        .collect::<Vec<_>>();
    let mut node = json!({
        "name": name,
        "translation": translation,
        "rotation": rotation,
        "scale": scale
    });
    if !children.is_empty() {
        node["children"] = json!(children);
    }
    nodes[index] = node;
    node_by_name.insert(name.clone(), index);
    node_by_name
        .entry(without_numeric_suffix(&name).to_owned())
        .or_insert(index);
    if let Some(matrix) = bone.get("InvBindMatrixInSkeletonSpace") {
        inverse_bind_by_name.insert(name, matrix16(Some(matrix)));
    }
    Some(index)
}

fn matrix16(value: Option<&Value>) -> [f32; 16] {
    let values = vec_from(
        value,
        16,
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    values.try_into().unwrap()
}

fn multiply_matrices(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            for k in 0..4 {
                out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
            }
        }
    }
    out
}

fn contains_gltf_scene_root(value: &Value) -> bool {
    if value
        .get("osg.MatrixTransform")
        .and_then(|transform| transform.get("Name"))
        .and_then(Value::as_str)
        == Some("GLTF_SceneRootNode")
    {
        return true;
    }
    match value {
        Value::Object(object) => object.values().any(contains_gltf_scene_root),
        Value::Array(array) => array.iter().any(contains_gltf_scene_root),
        _ => false,
    }
}

fn contains_texture_attributes(value: &Value) -> bool {
    if value.get("TextureAttributeList").is_some() {
        return true;
    }
    match value {
        Value::Object(object) => object.values().any(contains_texture_attributes),
        Value::Array(array) => array.iter().any(contains_texture_attributes),
        _ => false,
    }
}

fn scene_coordinate_matrix(scene: &Value) -> [f32; 16] {
    if contains_gltf_scene_root(scene) {
        matrix16(None)
    } else {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }
}

fn invert_affine_matrix(matrix: &[f32; 16]) -> Option<[f32; 16]> {
    let (a00, a01, a02) = (matrix[0], matrix[4], matrix[8]);
    let (a10, a11, a12) = (matrix[1], matrix[5], matrix[9]);
    let (a20, a21, a22) = (matrix[2], matrix[6], matrix[10]);
    let determinant = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20);
    if determinant.abs() < 1e-12 {
        return None;
    }
    let d = determinant.recip();
    let mut inverse = [
        (a11 * a22 - a12 * a21) * d,
        (a12 * a20 - a10 * a22) * d,
        (a10 * a21 - a11 * a20) * d,
        0.0,
        (a02 * a21 - a01 * a22) * d,
        (a00 * a22 - a02 * a20) * d,
        (a01 * a20 - a00 * a21) * d,
        0.0,
        (a01 * a12 - a02 * a11) * d,
        (a02 * a10 - a00 * a12) * d,
        (a00 * a11 - a01 * a10) * d,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ];
    let translation = [matrix[12], matrix[13], matrix[14]];
    for row in 0..3 {
        inverse[12 + row] = -(inverse[row] * translation[0]
            + inverse[4 + row] * translation[1]
            + inverse[8 + row] * translation[2]);
    }
    Some(inverse)
}

fn decompose_matrix(matrix: &[f32; 16]) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let translation = [matrix[12], matrix[13], matrix[14]];
    let mut scale = [
        (matrix[0] * matrix[0] + matrix[1] * matrix[1] + matrix[2] * matrix[2]).sqrt(),
        (matrix[4] * matrix[4] + matrix[5] * matrix[5] + matrix[6] * matrix[6]).sqrt(),
        (matrix[8] * matrix[8] + matrix[9] * matrix[9] + matrix[10] * matrix[10]).sqrt(),
    ];
    for value in &mut scale {
        if value.abs() < 1e-8 {
            *value = 1.0;
        }
    }
    let determinant = matrix[0] * (matrix[5] * matrix[10] - matrix[9] * matrix[6])
        - matrix[4] * (matrix[1] * matrix[10] - matrix[9] * matrix[2])
        + matrix[8] * (matrix[1] * matrix[6] - matrix[5] * matrix[2]);
    if determinant < 0.0 {
        scale[0] = -scale[0];
    }
    let m00 = matrix[0] / scale[0];
    let m01 = matrix[4] / scale[1];
    let m02 = matrix[8] / scale[2];
    let m10 = matrix[1] / scale[0];
    let m11 = matrix[5] / scale[1];
    let m12 = matrix[9] / scale[2];
    let m20 = matrix[2] / scale[0];
    let m21 = matrix[6] / scale[1];
    let m22 = matrix[10] / scale[2];
    let trace = m00 + m11 + m22;
    let rotation = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s * 0.25]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [s * 0.25, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(m01 + m10) / s, s * 0.25, (m12 + m21) / s, (m02 - m20) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(m02 + m20) / s, (m12 + m21) / s, s * 0.25, (m10 - m01) / s]
    };
    (translation, rotation, scale)
}

fn collect_skeleton_nodes(
    value: &Value,
    parent_matrix: &[f32; 16],
    nodes: &mut Vec<Value>,
    scene_roots: &mut Vec<usize>,
    node_by_name: &mut HashMap<String, usize>,
    inverse_bind_by_name: &mut HashMap<String, [f32; 16]>,
    seen: &mut HashSet<u64>,
) {
    if let Some(skeleton) = value.get("osgAnimation.Skeleton") {
        let id = skeleton.get("UniqueID").and_then(Value::as_u64);
        if id.is_none_or(|id| seen.insert(id)) {
            let children = skeleton
                .get("Children")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|child| add_bone_node(child, nodes, node_by_name, inverse_bind_by_name))
                .collect::<Vec<_>>();
            let root_index = nodes.len();
            let mut root = json!({
                "name": skeleton
                    .get("Name")
                    .and_then(Value::as_str)
                    .unwrap_or("Skeleton"),
                "matrix": parent_matrix
            });
            if !children.is_empty() {
                root["children"] = json!(children);
            }
            nodes.push(root);
            scene_roots.push(root_index);
        }
        return;
    }
    if let Some(transform) = value.get("osg.MatrixTransform") {
        let local = matrix16(transform.get("Matrix"));
        let combined =
            if transform.get("Name").and_then(Value::as_str) == Some("GLTF_SceneRootNode") {
                *parent_matrix
            } else {
                multiply_matrices(parent_matrix, &local)
            };
        if let Some(children) = transform.get("Children").and_then(Value::as_array) {
            for child in children {
                collect_skeleton_nodes(
                    child,
                    &combined,
                    nodes,
                    scene_roots,
                    node_by_name,
                    inverse_bind_by_name,
                    seen,
                );
            }
        }
        return;
    }
    match value {
        Value::Object(object) => {
            for child in object.values() {
                collect_skeleton_nodes(
                    child,
                    parent_matrix,
                    nodes,
                    scene_roots,
                    node_by_name,
                    inverse_bind_by_name,
                    seen,
                );
            }
        }
        Value::Array(array) => {
            for child in array {
                collect_skeleton_nodes(
                    child,
                    parent_matrix,
                    nodes,
                    scene_roots,
                    node_by_name,
                    inverse_bind_by_name,
                    seen,
                );
            }
        }
        _ => {}
    }
}

fn collect_geometries(
    osgjs: &Value,
    poly_bin: &[u8],
    wire_bin: Option<&[u8]>,
) -> Result<Vec<Geometry>> {
    let root = resolved_scene(osgjs);
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    traverse_geometries(
        &root,
        poly_bin,
        wire_bin,
        &scene_coordinate_matrix(&root),
        None,
        None,
        &mut out,
        &mut seen,
    )?;
    Ok(out)
}

fn add_rig_attributes(geometry: &mut Geometry, rig: &Value, poly_bin: &[u8]) -> Result<()> {
    let Some(attributes) = rig.get("VertexAttributeList").and_then(Value::as_object) else {
        return Ok(());
    };
    for (source_name, target_name, component_type) in
        [("Bones", "JOINTS_0", 5123), ("Weights", "WEIGHTS_0", 5126)]
    {
        let Some(definition) = attributes.get(source_name) else {
            continue;
        };
        let Some((array_type, array_definition)) =
            definition.get("Array").and_then(first_key_value)
        else {
            continue;
        };
        let item_size = definition
            .get("ItemSize")
            .and_then(Value::as_u64)
            .unwrap_or(4) as usize;
        let count = array_definition
            .get("Size")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let raw = read_buffer_array(poly_bin, array_definition, item_size, array_type)?;
        let data = if source_name == "Bones" {
            AttrData::U16(
                raw.as_i64_vec()
                    .into_iter()
                    .map(|value| value as u16)
                    .collect(),
            )
        } else {
            AttrData::F32(raw.to_f32_vec())
        };
        geometry.attributes.insert(
            target_name.to_owned(),
            Attribute {
                data,
                item_size,
                count,
                component_type,
                normalized: false,
            },
        );
    }
    let weights = geometry
        .attributes
        .get("WEIGHTS_0")
        .map(|attribute| attribute.data.to_f32_vec());
    if let (Some(weights), Some(joints)) = (weights, geometry.attributes.get_mut("JOINTS_0"))
        && let AttrData::U16(joint_values) = &mut joints.data
    {
        for (joint, weight) in joint_values.iter_mut().zip(weights) {
            if weight == 0.0 {
                *joint = 0;
            }
        }
    }

    if let Some(bone_map) = rig.get("BoneMap").and_then(Value::as_object) {
        let max_index = bone_map
            .values()
            .filter_map(Value::as_u64)
            .max()
            .unwrap_or(0) as usize;
        let mut names = vec![String::new(); max_index + 1];
        for (name, index) in bone_map {
            if let Some(slot) = index.as_u64().and_then(|i| names.get_mut(i as usize)) {
                *slot = name.clone();
            }
        }
        geometry.joint_names = names;
    }
    Ok(())
}

fn traverse_geometries(
    v: &Value,
    poly_bin: &[u8],
    wire_bin: Option<&[u8]>,
    parent_matrix: &[f32; 16],
    skeleton_matrix: Option<[f32; 16]>,
    animation_target: Option<String>,
    out: &mut Vec<Geometry>,
    seen: &mut HashSet<u64>,
) -> Result<()> {
    if let Some(transform) = v.get("osg.MatrixTransform") {
        let local = matrix16(transform.get("Matrix"));
        let combined =
            if transform.get("Name").and_then(Value::as_str) == Some("GLTF_SceneRootNode") {
                *parent_matrix
            } else {
                multiply_matrices(parent_matrix, &local)
            };
        let target = transform
            .get("UpdateCallbacks")
            .and_then(Value::as_array)
            .and_then(|callbacks| {
                callbacks.iter().find_map(|callback| {
                    callback
                        .pointer("/osgAnimation.UpdateMatrixTransform/Name")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
            })
            .or(animation_target);
        if let Some(children) = transform.get("Children").and_then(Value::as_array) {
            for child in children {
                traverse_geometries(
                    child,
                    poly_bin,
                    wire_bin,
                    &combined,
                    skeleton_matrix,
                    target.clone(),
                    out,
                    seen,
                )?;
            }
        }
        return Ok(());
    }
    if let Some(skeleton) = v.get("osgAnimation.Skeleton") {
        if let Some(children) = skeleton.get("Children").and_then(Value::as_array) {
            for child in children {
                traverse_geometries(
                    child,
                    poly_bin,
                    wire_bin,
                    parent_matrix,
                    Some(*parent_matrix),
                    animation_target.clone(),
                    out,
                    seen,
                )?;
            }
        }
        return Ok(());
    }
    if let Some(rig) = v.get("osgAnimation.RigGeometry") {
        if let Some(source) = rig.get("SourceGeometry").and_then(|source| {
            source
                .get("osg.Geometry")
                .or_else(|| source.get("osgAnimation.MorphGeometry"))
        }) {
            let id = source.get("UniqueID").and_then(Value::as_u64);
            if id.is_none_or(|id| seen.insert(id)) {
                match process_geometry(source, poly_bin, wire_bin) {
                    Ok(Some(mut geometry)) => {
                        add_rig_attributes(&mut geometry, rig, poly_bin)?;
                        geometry.matrix = *parent_matrix;
                        geometry.skeleton_matrix = skeleton_matrix;
                        geometry.animation_target = animation_target.clone();
                        out.push(geometry);
                    }
                    Ok(None) => {}
                    Err(error) => eprintln!("  Warning: skipping rig geometry: {error}"),
                }
            }
        }
        return Ok(());
    }
    if let Some(geom) = v.get("osg.Geometry") {
        if let Some(id) = geom.get("UniqueID").and_then(Value::as_u64) {
            if !seen.insert(id) {
                return Ok(());
            }
        }
        match process_geometry(geom, poly_bin, wire_bin) {
            Ok(Some(mut geometry)) => {
                geometry.matrix = *parent_matrix;
                geometry.animation_target = animation_target.clone();
                out.push(geometry);
            }
            Ok(None) => {}
            Err(e) => eprintln!("  Warning: skipping geometry: {e}"),
        }
    }
    for ptr in [
        "/osg.Node/Children",
        "/osg.MatrixTransform/Children",
        "/osgAnimation.Skeleton/Children",
        "/Children",
    ] {
        if let Some(children) = v.pointer(ptr).and_then(Value::as_array) {
            for child in children {
                traverse_geometries(
                    child,
                    poly_bin,
                    wire_bin,
                    parent_matrix,
                    skeleton_matrix,
                    animation_target.clone(),
                    out,
                    seen,
                )?;
            }
        }
    }
    Ok(())
}

fn push_padded(bin: &mut Vec<u8>, bytes: &[u8]) -> (usize, usize) {
    let offset = bin.len();
    bin.extend_from_slice(bytes);
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    (offset, bytes.len())
}

fn bytes_for_attr(attr: &Attribute) -> Vec<u8> {
    let mut out = Vec::new();
    match &attr.data {
        AttrData::F32(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes())
            }
        }
        AttrData::I32(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes())
            }
        }
        AttrData::U32(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes())
            }
        }
        AttrData::U16(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes())
            }
        }
        AttrData::U8(v) => out.extend_from_slice(v),
        AttrData::I16(v) => {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes())
            }
        }
    }
    out
}

fn add_accessor(gltf: &mut Value, bin: &mut Vec<u8>, attr: &Attribute) -> Result<usize> {
    let bytes = bytes_for_attr(attr);
    let (offset, len) = push_padded(bin, &bytes);
    let bv_idx = gltf["bufferViews"].as_array().unwrap().len();
    gltf["bufferViews"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "buffer": 0, "byteOffset": offset, "byteLength": len }));
    let vals = attr.data.to_f32_vec();
    let mut min = vec![f32::INFINITY; attr.item_size];
    let mut max = vec![f32::NEG_INFINITY; attr.item_size];
    for i in 0..attr.count {
        for j in 0..attr.item_size {
            if let Some(v) = vals.get(i * attr.item_size + j) {
                min[j] = min[j].min(*v);
                max[j] = max[j].max(*v);
            }
        }
    }
    let mut acc = json!({
        "bufferView": bv_idx,
        "byteOffset": 0,
        "componentType": attr.component_type,
        "count": attr.count,
        "type": match attr.item_size { 1 => "SCALAR", 2 => "VEC2", 3 => "VEC3", 4 => "VEC4", _ => "SCALAR" },
        "min": min,
        "max": max
    });
    if attr.normalized {
        acc["normalized"] = Value::Bool(true);
    }
    let idx = gltf["accessors"].as_array().unwrap().len();
    gltf["accessors"].as_array_mut().unwrap().push(acc);
    Ok(idx)
}

fn set_accessor_target(gltf: &mut Value, accessor: usize, target: u32) {
    if let Some(buffer_view) = gltf["accessors"][accessor]["bufferView"].as_u64() {
        gltf["bufferViews"][buffer_view as usize]["target"] = json!(target);
    }
}

fn add_mat4_accessor(gltf: &mut Value, bin: &mut Vec<u8>, matrices: &[f32]) -> usize {
    let mut bytes = Vec::with_capacity(matrices.len() * 4);
    for value in matrices {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let (offset, length) = push_padded(bin, &bytes);
    let buffer_view = gltf["bufferViews"].as_array().unwrap().len();
    gltf["bufferViews"].as_array_mut().unwrap().push(json!({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": length
    }));
    let accessor = gltf["accessors"].as_array().unwrap().len();
    gltf["accessors"].as_array_mut().unwrap().push(json!({
        "bufferView": buffer_view,
        "byteOffset": 0,
        "componentType": 5126,
        "count": matrices.len() / 16,
        "type": "MAT4"
    }));
    accessor
}

fn is_environment_geometry(
    geometry: &Geometry,
    materials: &HashMap<String, MaterialEntry>,
) -> bool {
    if !geometry.joint_names.is_empty() {
        return false;
    }
    let Some(position) = geometry.attributes.get("POSITION") else {
        return false;
    };
    if position.count > 128 || position.item_size < 3 {
        return false;
    }
    let Some(material) = geometry
        .material_name
        .as_ref()
        .and_then(|name| materials.get(name))
    else {
        return false;
    };
    if material.base_color_texture.is_some() || material.emissive_texture.is_some() {
        return false;
    }
    let AttrData::F32(values) = &position.data else {
        return false;
    };
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for vertex in values.chunks_exact(position.item_size) {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(vertex[axis]);
            maximum[axis] = maximum[axis].max(vertex[axis]);
        }
    }
    let mut extents = [
        maximum[0] - minimum[0],
        maximum[1] - minimum[1],
        maximum[2] - minimum[2],
    ];
    extents.sort_by(f32::total_cmp);
    extents[2] > 0.0 && extents[0] <= extents[2] * 0.01 && extents[1] >= extents[2] * 0.5
}

fn add_image(gltf: &mut Value, bin: &mut Vec<u8>, path: &Path) -> Result<Option<usize>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    let mime = if path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("png"))
    {
        "image/png"
    } else {
        "image/jpeg"
    };
    Ok(Some(add_image_bytes(gltf, bin, &bytes, mime)))
}

fn add_image_bytes(gltf: &mut Value, bin: &mut Vec<u8>, bytes: &[u8], mime: &str) -> usize {
    let (offset, len) = push_padded(bin, &bytes);
    let bv_idx = gltf["bufferViews"].as_array().unwrap().len();
    gltf["bufferViews"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "buffer": 0, "byteOffset": offset, "byteLength": len }));
    let image_idx = gltf["images"].as_array().unwrap().len();
    gltf["images"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "bufferView": bv_idx, "mimeType": mime }));
    image_idx
}

fn add_sampler(gltf: &mut Value, settings: SamplerSettings) -> usize {
    let index = gltf["samplers"].as_array().unwrap().len();
    gltf["samplers"].as_array_mut().unwrap().push(json!({
        "magFilter": settings.mag_filter,
        "minFilter": settings.min_filter,
        "wrapS": settings.wrap_s,
        "wrapT": settings.wrap_t
    }));
    index
}

fn add_texture(gltf: &mut Value, image: usize, sampler: usize) -> usize {
    let index = gltf["textures"].as_array().unwrap().len();
    gltf["textures"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "source": image, "sampler": sampler }));
    index
}

fn texture_info(
    usage: &TextureUse,
    texture_indices: &HashMap<(String, SamplerSettings), usize>,
    texcoord_units: &HashMap<u32, u32>,
    uses_texture_transform: &mut bool,
    uvs_flipped: bool,
) -> Option<Value> {
    let index = texture_indices
        .get(&(usage.uid.clone(), usage.sampler))
        .copied()?;
    Some(texture_info_for_index(
        index,
        usage,
        texcoord_units,
        uses_texture_transform,
        uvs_flipped,
    ))
}

fn texture_info_for_index(
    index: usize,
    usage: &TextureUse,
    texcoord_units: &HashMap<u32, u32>,
    uses_texture_transform: &mut bool,
    uvs_flipped: bool,
) -> Value {
    let texcoord = texcoord_units
        .get(&usage.texcoord_unit)
        .copied()
        .or_else(|| texcoord_units.values().copied().min())
        .unwrap_or(0);
    let transform = &usage.transform;
    let (rotation, offset) = if uvs_flipped {
        let sin = transform.rotation.sin();
        let cos = transform.rotation.cos();
        (
            -transform.rotation,
            [
                transform.offset[0] - sin * transform.scale[1],
                1.0 - transform.offset[1] - cos * transform.scale[1],
            ],
        )
    } else {
        (transform.rotation, transform.offset)
    };
    let changed = offset[0].abs() > 1e-6
        || offset[1].abs() > 1e-6
        || (transform.scale[0] - 1.0).abs() > 1e-6
        || (transform.scale[1] - 1.0).abs() > 1e-6
        || rotation.abs() > 1e-6;
    let mut info = json!({ "index": index, "texCoord": texcoord });
    if changed {
        *uses_texture_transform = true;
        let mut extension = json!({
            "offset": offset,
            "scale": transform.scale
        });
        if rotation.abs() > 1e-6 {
            extension["rotation"] = json!(rotation);
        }
        info["extensions"] = json!({ "KHR_texture_transform": extension });
    }
    info
}

fn texture_file<'a>(texture: &'a TextureEntry, texture_dir: &Path) -> PathBuf {
    texture_dir.join(texture.clean_file.as_ref().unwrap_or(&texture.filename))
}

fn add_metallic_roughness_texture(
    gltf: &mut Value,
    bin: &mut Vec<u8>,
    texture_dir: &Path,
    metallic: Option<&TextureEntry>,
    roughness: Option<&TextureEntry>,
    invert_roughness: bool,
) -> Result<Option<usize>> {
    let load = |texture: Option<&TextureEntry>| -> Result<Option<image::GrayImage>> {
        let Some(texture) = texture else {
            return Ok(None);
        };
        let path = texture_file(texture, texture_dir);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(image::open(path)?.to_luma8()))
    };
    let mut metallic = load(metallic)?;
    let mut roughness = load(roughness)?;
    let Some((width, height)) = metallic
        .as_ref()
        .map(|image| image.dimensions())
        .or_else(|| roughness.as_ref().map(|image| image.dimensions()))
    else {
        return Ok(None);
    };
    if metallic
        .as_ref()
        .is_some_and(|image| image.dimensions() != (width, height))
    {
        metallic = metallic
            .map(|image| image::imageops::resize(&image, width, height, FilterType::Triangle));
    }
    if roughness
        .as_ref()
        .is_some_and(|image| image.dimensions() != (width, height))
    {
        roughness = roughness
            .map(|image| image::imageops::resize(&image, width, height, FilterType::Triangle));
    }
    let packed = ImageBuffer::from_fn(width, height, |x, y| {
        let metal = metallic
            .as_ref()
            .map_or(255, |image| image.get_pixel(x, y)[0]);
        let rough = roughness
            .as_ref()
            .map_or(255, |image| image.get_pixel(x, y)[0]);
        Rgba([
            255,
            if invert_roughness { 255 - rough } else { rough },
            metal,
            255,
        ])
    });
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(packed).write_to(&mut bytes, ImageFormat::Png)?;
    Ok(Some(add_image_bytes(
        gltf,
        bin,
        bytes.get_ref(),
        "image/png",
    )))
}

fn same_texture_mapping(left: &TextureUse, right: &TextureUse) -> bool {
    left.texcoord_unit == right.texcoord_unit
        && left.transform.offset == right.transform.offset
        && left.transform.scale == right.transform.scale
        && left.transform.rotation == right.transform.rotation
}

fn add_base_alpha_texture(
    gltf: &mut Value,
    bin: &mut Vec<u8>,
    texture_dir: &Path,
    base: Option<&TextureEntry>,
    alpha: &TextureEntry,
    invert: bool,
    alpha_channel: bool,
) -> Result<Option<usize>> {
    let alpha_path = texture_file(alpha, texture_dir);
    if !alpha_path.exists() {
        return Ok(None);
    }
    let alpha_image = image::open(alpha_path)?;
    let alpha = if alpha_channel && alpha_image.color().has_alpha() {
        let rgba = alpha_image.to_rgba8();
        ImageBuffer::from_fn(rgba.width(), rgba.height(), |x, y| {
            image::Luma([rgba.get_pixel(x, y)[3]])
        })
    } else {
        alpha_image.to_luma8()
    };
    let mut packed = base
        .map(|texture| texture_file(texture, texture_dir))
        .filter(|path| path.exists())
        .map(image::open)
        .transpose()?
        .map(|image| image.to_rgba8())
        .unwrap_or_else(|| {
            ImageBuffer::from_pixel(alpha.width(), alpha.height(), Rgba([255, 255, 255, 255]))
        });
    let (width, height) = packed.dimensions();
    let alpha = if alpha.dimensions() == (width, height) {
        alpha
    } else {
        image::imageops::resize(&alpha, width, height, FilterType::Triangle)
    };
    for (pixel, alpha) in packed.pixels_mut().zip(alpha.pixels()) {
        let alpha = if invert { 255 - alpha[0] } else { alpha[0] };
        pixel[3] = ((pixel[3] as u16 * alpha as u16) / 255) as u8;
    }
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(packed).write_to(&mut bytes, ImageFormat::Png)?;
    Ok(Some(add_image_bytes(
        gltf,
        bin,
        bytes.get_ref(),
        "image/png",
    )))
}

fn add_flipped_normal_texture(
    gltf: &mut Value,
    bin: &mut Vec<u8>,
    texture_dir: &Path,
    texture: &TextureEntry,
) -> Result<Option<usize>> {
    let path = texture_file(texture, texture_dir);
    if !path.exists() {
        return Ok(None);
    }
    let mut image = image::open(path)?.to_rgba8();
    for pixel in image.pixels_mut() {
        pixel[1] = 255 - pixel[1];
    }
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageFormat::Png)?;
    Ok(Some(add_image_bytes(
        gltf,
        bin,
        bytes.get_ref(),
        "image/png",
    )))
}

#[allow(clippy::too_many_arguments)]
fn add_material_variant(
    source: &MaterialEntry,
    texcoord_units: &HashMap<u32, u32>,
    texture_indices: &HashMap<(String, SamplerSettings), usize>,
    texture_files: &HashMap<String, TextureEntry>,
    texture_dir: &Path,
    sampler_indices: &mut HashMap<SamplerSettings, usize>,
    packed_indices: &mut HashMap<(Option<String>, Option<String>, bool, SamplerSettings), usize>,
    alpha_indices: &mut HashMap<(Option<String>, String, bool, bool, SamplerSettings), usize>,
    normal_indices: &mut HashMap<(String, bool, SamplerSettings), usize>,
    uses_texture_transform: &mut bool,
    uvs_flipped: bool,
    uses_unlit: &mut bool,
    used_material_extensions: &mut HashSet<String>,
    gltf: &mut Value,
    bin: &mut Vec<u8>,
) -> Result<usize> {
    let mut material = json!({
        "name": source.name,
        "doubleSided": source.double_sided,
        "alphaMode": source.alpha_mode,
        "pbrMetallicRoughness": {
            "baseColorFactor": source.base_color,
            "metallicFactor": source.metallic_factor,
            "roughnessFactor": source.roughness_factor
        }
    });
    let alpha_usage = source
        .alpha_mask_texture
        .as_ref()
        .or(source.opacity_texture.as_ref());
    let combined_base = alpha_usage
        .filter(|alpha| {
            source
                .base_color_texture
                .as_ref()
                .is_none_or(|base| same_texture_mapping(base, alpha))
        })
        .and_then(|alpha| {
            let base_uid = source
                .base_color_texture
                .as_ref()
                .map(|base| base.uid.clone());
            let sampler = source
                .base_color_texture
                .as_ref()
                .map_or(alpha.sampler, |base| base.sampler);
            let key = (
                base_uid.clone(),
                alpha.uid.clone(),
                source.alpha_invert,
                alpha.alpha_channel,
                sampler,
            );
            if let Some(index) = alpha_indices.get(&key) {
                return Some((*index, sampler));
            }
            let image = add_base_alpha_texture(
                gltf,
                bin,
                texture_dir,
                base_uid.as_ref().and_then(|uid| texture_files.get(uid)),
                texture_files.get(&alpha.uid)?,
                source.alpha_invert,
                alpha.alpha_channel,
            )
            .ok()
            .flatten()?;
            let sampler_index = *sampler_indices
                .entry(sampler)
                .or_insert_with(|| add_sampler(gltf, sampler));
            let index = add_texture(gltf, image, sampler_index);
            alpha_indices.insert(key, index);
            Some((index, sampler))
        });
    let base_info = if let Some((index, sampler)) = combined_base {
        let usage = source.base_color_texture.as_ref().or(alpha_usage).unwrap();
        let mut usage = usage.clone();
        usage.sampler = sampler;
        Some(texture_info_for_index(
            index,
            &usage,
            texcoord_units,
            uses_texture_transform,
            uvs_flipped,
        ))
    } else {
        source.base_color_texture.as_ref().and_then(|usage| {
            texture_info(
                usage,
                texture_indices,
                texcoord_units,
                uses_texture_transform,
                uvs_flipped,
            )
        })
    };
    if let Some(info) = base_info {
        material["pbrMetallicRoughness"]["baseColorTexture"] = info;
    }
    if source.emissive_enabled {
        let strength = source.emissive_color.iter().copied().fold(1.0f32, f32::max);
        material["emissiveFactor"] = json!([
            source.emissive_color[0] / strength,
            source.emissive_color[1] / strength,
            source.emissive_color[2] / strength
        ]);
        if strength > 1.0 {
            material["extensions"]["KHR_materials_emissive_strength"] =
                json!({ "emissiveStrength": strength });
            used_material_extensions.insert("KHR_materials_emissive_strength".to_owned());
        }
    }
    if let Some(info) = source.emissive_texture.as_ref().and_then(|usage| {
        texture_info(
            usage,
            texture_indices,
            texcoord_units,
            uses_texture_transform,
            uvs_flipped,
        )
    }) {
        material["emissiveTexture"] = info;
    }
    if let Some(mut info) = source.occlusion_texture.as_ref().and_then(|usage| {
        texture_info(
            usage,
            texture_indices,
            texcoord_units,
            uses_texture_transform,
            uvs_flipped,
        )
    }) {
        info["strength"] = json!(1.0);
        material["occlusionTexture"] = info;
    }
    let normal_info = if source.normal_flip_y {
        source.normal_texture.as_ref().and_then(|usage| {
            let key = (usage.uid.clone(), true, usage.sampler);
            let index = if let Some(index) = normal_indices.get(&key) {
                Some(*index)
            } else {
                let image = add_flipped_normal_texture(
                    gltf,
                    bin,
                    texture_dir,
                    texture_files.get(&usage.uid)?,
                )
                .ok()
                .flatten()?;
                let sampler = *sampler_indices
                    .entry(usage.sampler)
                    .or_insert_with(|| add_sampler(gltf, usage.sampler));
                let index = add_texture(gltf, image, sampler);
                normal_indices.insert(key, index);
                Some(index)
            }?;
            Some(texture_info_for_index(
                index,
                usage,
                texcoord_units,
                uses_texture_transform,
                uvs_flipped,
            ))
        })
    } else {
        source.normal_texture.as_ref().and_then(|usage| {
            texture_info(
                usage,
                texture_indices,
                texcoord_units,
                uses_texture_transform,
                uvs_flipped,
            )
        })
    };
    if let Some(mut info) = normal_info {
        info["scale"] = json!(source.normal_scale);
        material["normalTexture"] = info;
    }
    if source.metallic_texture.is_some() || source.roughness_texture.is_some() {
        let usage = source
            .roughness_texture
            .as_ref()
            .or(source.metallic_texture.as_ref())
            .unwrap();
        let key = (
            source
                .metallic_texture
                .as_ref()
                .map(|texture| texture.uid.clone()),
            source
                .roughness_texture
                .as_ref()
                .map(|texture| texture.uid.clone()),
            source.roughness_invert,
            usage.sampler,
        );
        let packed_index = if let Some(index) = packed_indices.get(&key) {
            Some(*index)
        } else {
            let image = add_metallic_roughness_texture(
                gltf,
                bin,
                texture_dir,
                source
                    .metallic_texture
                    .as_ref()
                    .and_then(|texture| texture_files.get(&texture.uid)),
                source
                    .roughness_texture
                    .as_ref()
                    .and_then(|texture| texture_files.get(&texture.uid)),
                source.roughness_invert,
            )?;
            image.map(|image| {
                let sampler = *sampler_indices
                    .entry(usage.sampler)
                    .or_insert_with(|| add_sampler(gltf, usage.sampler));
                let index = add_texture(gltf, image, sampler);
                packed_indices.insert(key, index);
                index
            })
        };
        if let Some(index) = packed_index {
            material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = texture_info_for_index(
                index,
                usage,
                texcoord_units,
                uses_texture_transform,
                uvs_flipped,
            );
        }
    }
    if source.alpha_mode == "MASK" {
        material["alphaCutoff"] = json!(source.alpha_cutoff);
    }
    if !source.unlit && !source.extensions.is_empty() {
        for (name, extension) in &source.extensions {
            material["extensions"][name] = extension.clone();
            used_material_extensions.insert(name.clone());
        }
    }
    if source.unlit {
        *uses_unlit = true;
        if material.get("extensions").is_none() {
            material["extensions"] = json!({});
        }
        material["extensions"]["KHR_materials_unlit"] = json!({});
    }
    let index = gltf["materials"].as_array().unwrap().len();
    gltf["materials"].as_array_mut().unwrap().push(material);
    Ok(index)
}

fn user_data_values(value: &Value) -> HashMap<String, Value> {
    let mut result = HashMap::new();
    if let Some(values) = value
        .pointer("/UserDataContainer/Values")
        .and_then(Value::as_array)
    {
        for entry in values {
            let Some(name) = entry.get("Name").and_then(Value::as_str) else {
                continue;
            };
            let raw = entry.get("Value").cloned().unwrap_or(Value::Null);
            let parsed = raw
                .as_str()
                .and_then(|text| text.parse::<f64>().ok())
                .map(Value::from)
                .unwrap_or(raw);
            result.insert(name.to_owned(), parsed);
        }
    }
    result
}

fn unpack_components(values: &[f32], item_size: usize) -> Vec<f32> {
    let count = values.len() / item_size;
    let mut unpacked = vec![0.0; values.len()];
    for index in 0..count {
        for component in 0..item_size {
            unpacked[index * item_size + component] = values[index + count * component];
        }
    }
    unpacked
}

fn decode_quaternion_keys(encoded: &[f32], epsilon: f32, nphi: f32) -> Vec<f32> {
    let count = encoded.len() / 3;
    let mut output = vec![0.0; count * 4];
    let cos_epsilon = (epsilon * 0.01745329251).cos();
    let phi_step = std::f32::consts::PI / (nphi - 1.0);
    let gamma_step = 1.57079632679 / (nphi - 1.0);
    for index in 0..count {
        let s = encoded[index * 3];
        let x = encoded[index * 3 + 1];
        let phi = s * phi_step;
        let radial = phi.cos();
        let vertical = phi.sin();
        let next_phi = phi + gamma_step;
        let ratio = ((cos_epsilon - radial * next_phi.cos())
            / (1e-5f32.max(vertical * next_phi.sin())))
        .clamp(-1.0, 1.0);
        let azimuth = x * 6.28318530718 / (std::f32::consts::PI / 1e-5f32.max(ratio.acos())).ceil();
        let angle = encoded[index * 3 + 2] * 0.000047938362584151635;
        let sin_angle = angle.sin();
        output[index * 4] = sin_angle * vertical * azimuth.cos();
        output[index * 4 + 1] = sin_angle * vertical * azimuth.sin();
        output[index * 4 + 2] = sin_angle * radial;
        output[index * 4 + 3] = angle.cos();
    }
    output
}

fn cumulative_values(values: &mut [f32], item_size: usize) {
    for index in 1..values.len() / item_size {
        for component in 0..item_size {
            values[index * item_size + component] += values[(index - 1) * item_size + component];
        }
    }
}

fn cumulative_quaternions(values: &mut [f32]) {
    for index in 1..values.len() / 4 {
        let previous = (index - 1) * 4;
        let current = index * 4;
        let px = values[previous];
        let py = values[previous + 1];
        let pz = values[previous + 2];
        let pw = values[previous + 3];
        let cx = values[current];
        let cy = values[current + 1];
        let cz = values[current + 2];
        let cw = values[current + 3];
        values[current] = px * cw + py * cz - pz * cy + pw * cx;
        values[current + 1] = -px * cz + py * cw + pz * cx + pw * cy;
        values[current + 2] = px * cy - py * cx + pz * cw + pw * cz;
        values[current + 3] = -px * cx - py * cy - pz * cz + pw * cw;
    }
}

fn make_quaternions_continuous(values: &mut [f32]) {
    for index in 1..values.len() / 4 {
        let previous = (index - 1) * 4;
        let current = index * 4;
        let dot = values[previous] * values[current]
            + values[previous + 1] * values[current + 1]
            + values[previous + 2] * values[current + 2]
            + values[previous + 3] * values[current + 3];
        if dot < 0.0 {
            values[current] = -values[current];
            values[current + 1] = -values[current + 1];
            values[current + 2] = -values[current + 2];
            values[current + 3] = -values[current + 3];
        }
    }
}

fn deduplicate_keyframes(
    times: Vec<f32>,
    values: Vec<f32>,
    components: usize,
) -> (Vec<f32>, Vec<f32>) {
    if components == 0 || values.len() / components != times.len() {
        return (times, values);
    }
    let mut filtered_times = Vec::with_capacity(times.len());
    let mut filtered_values = Vec::with_capacity(values.len());
    for (index, time) in times.into_iter().enumerate() {
        if filtered_times.last().is_none_or(|last| time > *last) {
            filtered_times.push(time);
            filtered_values
                .extend_from_slice(&values[index * components..(index + 1) * components]);
        } else if filtered_times.last() == Some(&time) {
            let start = filtered_values.len() - components;
            filtered_values[start..]
                .copy_from_slice(&values[index * components..(index + 1) * components]);
        }
    }
    (filtered_times, filtered_values)
}

fn sample_scalar_track(track: &ScalarTrack, time: f32) -> f32 {
    if track.times.is_empty() {
        return 0.0;
    }
    match track
        .times
        .binary_search_by(|candidate| candidate.total_cmp(&time))
    {
        Ok(index) => track.values[index],
        Err(0) => track.values[0],
        Err(index) if index >= track.times.len() => *track.values.last().unwrap_or(&0.0),
        Err(index) => {
            let previous = index - 1;
            let span = track.times[index] - track.times[previous];
            if span <= f32::EPSILON {
                track.values[index]
            } else {
                let amount = (time - track.times[previous]) / span;
                track.values[previous] + (track.values[index] - track.values[previous]) * amount
            }
        }
    }
}

fn decode_animation_array(
    animation_bin: &[u8],
    definition: &Value,
    metadata: &HashMap<String, Value>,
    compressed: bool,
    packed: bool,
    components: usize,
) -> Result<Vec<f32>> {
    let Some((array_type, array_definition)) = definition.get("Array").and_then(first_key_value)
    else {
        bail!("animation array definition missing");
    };
    let item_size = definition
        .get("ItemSize")
        .and_then(Value::as_u64)
        .unwrap_or(components as u64) as usize;
    let mut values =
        read_buffer_array(animation_bin, array_definition, item_size, array_type)?.to_f32_vec();
    if packed && components != 1 {
        values = unpack_components(&values, item_size);
    }
    let mode = num(metadata.get("channel_mode")) as u32;
    if components == 4 && mode & 8 != 0 {
        let epsilon = meta_value(metadata, "epsilon");
        let nphi = meta_value(metadata, "nphi");
        values = decode_quaternion_keys(
            &values,
            if epsilon == 0.0 { 0.25 } else { epsilon },
            if nphi == 0.0 { 720.0 } else { nphi },
        );
        if compressed && mode & 4 != 0 {
            let mut with_origin = vec![
                meta_value(metadata, "ox"),
                meta_value(metadata, "oy"),
                meta_value(metadata, "oz"),
                meta_value(metadata, "ow"),
            ];
            with_origin.extend(values);
            cumulative_quaternions(&mut with_origin);
            values = with_origin;
        }
        return Ok(values);
    }
    if compressed && components == 3 && mode & 1 != 0 {
        values = dequantize(
            &values,
            &[
                meta_value(metadata, "bx"),
                meta_value(metadata, "by"),
                meta_value(metadata, "bz"),
            ],
            &[
                meta_value(metadata, "hx"),
                meta_value(metadata, "hy"),
                meta_value(metadata, "hz"),
            ],
            3,
        );
    }
    let delta_encoded = compressed
        && if components == 1 {
            mode & 16 != 0
        } else {
            mode & 4 != 0
        };
    if delta_encoded {
        let mut with_origin = if components == 3 {
            vec![
                meta_value(metadata, "ox"),
                meta_value(metadata, "oy"),
                meta_value(metadata, "oz"),
            ]
        } else {
            vec![meta_value(metadata, "ot")]
        };
        with_origin.extend(values);
        cumulative_values(&mut with_origin, item_size);
        values = with_origin;
    }
    Ok(values)
}

fn export_animation(
    animation: &Value,
    animation_bin: &[u8],
    node_by_name: &HashMap<String, usize>,
    morph_bindings: &HashMap<String, Vec<MorphTargetBinding>>,
    gltf: &mut Value,
    bin: &mut Vec<u8>,
) -> Result<usize> {
    let mut samplers = Vec::new();
    let mut channels = Vec::new();
    let mut morph_groups: HashMap<usize, (usize, Vec<Option<ScalarTrack>>)> = HashMap::new();
    let Some(source_channels) = animation.get("Channels").and_then(Value::as_array) else {
        return Ok(0);
    };
    for wrapper in source_channels {
        let Some((channel_type, channel)) = first_key_value(wrapper) else {
            continue;
        };
        let Some(target_name) = channel.get("TargetName").and_then(Value::as_str) else {
            continue;
        };
        let Some(channel_name) = channel.get("Name").and_then(Value::as_str) else {
            continue;
        };
        if channel_type.contains("Float") {
            let Some(bindings) = morph_bindings.get(target_name) else {
                continue;
            };
            let metadata = user_data_values(channel);
            let compressed = channel_type.contains("Compressed");
            let Some(key_frames) = channel.get("KeyFrames") else {
                continue;
            };
            let times = decode_animation_array(
                animation_bin,
                &key_frames["Time"],
                &metadata,
                compressed,
                false,
                1,
            )?;
            let values = decode_animation_array(
                animation_bin,
                &key_frames["Key"],
                &metadata,
                compressed,
                channel_type.contains("Packed"),
                1,
            )?;
            let (times, values) = deduplicate_keyframes(times, values, 1);
            if times.is_empty() || values.len() != times.len() {
                continue;
            }
            let track = ScalarTrack { times, values };
            for binding in bindings {
                let group = morph_groups
                    .entry(binding.node)
                    .or_insert_with(|| (binding.target_count, vec![None; binding.target_count]));
                if group.0 == binding.target_count && binding.target_index < group.1.len() {
                    group.1[binding.target_index] = Some(track.clone());
                }
            }
            continue;
        }
        let Some(node) = node_by_name
            .get(target_name)
            .or_else(|| node_by_name.get(without_numeric_suffix(target_name)))
            .copied()
        else {
            continue;
        };
        let (path, components) = match channel_name {
            "translate" => ("translation", 3),
            "rotate" => ("rotation", 4),
            "scale" => ("scale", 3),
            _ => continue,
        };
        let metadata = user_data_values(channel);
        let compressed = channel_type.contains("Compressed");
        let packed = channel_type.contains("Packed");
        let Some(key_frames) = channel.get("KeyFrames") else {
            continue;
        };
        let mut times = decode_animation_array(
            animation_bin,
            &key_frames["Time"],
            &metadata,
            compressed,
            false,
            1,
        )?;
        let mut values = decode_animation_array(
            animation_bin,
            &key_frames["Key"],
            &metadata,
            compressed,
            packed,
            components,
        )?;
        (times, values) = deduplicate_keyframes(times, values, components);
        if path == "rotation" {
            make_quaternions_continuous(&mut values);
        }
        if times.is_empty() || values.len() / components != times.len() {
            continue;
        }
        let input = add_accessor(
            gltf,
            bin,
            &Attribute {
                data: AttrData::F32(times.clone()),
                item_size: 1,
                count: times.len(),
                component_type: 5126,
                normalized: false,
            },
        )?;
        let output = add_accessor(
            gltf,
            bin,
            &Attribute {
                data: AttrData::F32(values),
                item_size: components,
                count: times.len(),
                component_type: 5126,
                normalized: false,
            },
        )?;
        let sampler = samplers.len();
        samplers.push(json!({
            "input": input,
            "output": output,
            "interpolation": "LINEAR"
        }));
        channels.push(json!({
            "sampler": sampler,
            "target": {
                "node": node,
                "path": path
            }
        }));
    }
    let mut morph_nodes = morph_groups.into_iter().collect::<Vec<_>>();
    morph_nodes.sort_by_key(|(node, _)| *node);
    for (node, (target_count, tracks)) in morph_nodes {
        let mut times = tracks
            .iter()
            .flatten()
            .flat_map(|track| track.times.iter().copied())
            .collect::<Vec<_>>();
        times.sort_by(f32::total_cmp);
        times.dedup();
        if times.is_empty() {
            continue;
        }
        let mut weights = Vec::with_capacity(times.len() * target_count);
        for time in &times {
            for track in &tracks {
                weights.push(
                    track
                        .as_ref()
                        .map_or(0.0, |track| sample_scalar_track(track, *time)),
                );
            }
        }
        let input = add_accessor(
            gltf,
            bin,
            &Attribute {
                data: AttrData::F32(times.clone()),
                item_size: 1,
                count: times.len(),
                component_type: 5126,
                normalized: false,
            },
        )?;
        let output = add_accessor(
            gltf,
            bin,
            &Attribute {
                data: AttrData::F32(weights),
                item_size: 1,
                count: times.len() * target_count,
                component_type: 5126,
                normalized: false,
            },
        )?;
        let sampler = samplers.len();
        samplers.push(json!({
            "input": input,
            "output": output,
            "interpolation": "LINEAR"
        }));
        channels.push(json!({
            "sampler": sampler,
            "target": {
                "node": node,
                "path": "weights"
            }
        }));
    }
    if channels.is_empty() {
        return Ok(0);
    }
    gltf["animations"].as_array_mut().unwrap().push(json!({
        "name": animation
            .get("Name")
            .and_then(Value::as_str)
            .unwrap_or("Animation"),
        "samplers": samplers,
        "channels": channels
    }));
    Ok(channels.len())
}

fn export_animations_from_scene(
    value: &Value,
    animation_bins: &HashMap<String, Vec<u8>>,
    node_by_name: &HashMap<String, usize>,
    morph_bindings: &HashMap<String, Vec<MorphTargetBinding>>,
    gltf: &mut Value,
    bin: &mut Vec<u8>,
) -> Result<usize> {
    if let Some(animation) = value.get("osgAnimation.Animation") {
        let name = animation
            .get("Name")
            .and_then(Value::as_str)
            .unwrap_or("Animation")
            .to_ascii_lowercase();
        let animation_bin = animation_bins.get(&name).or_else(|| {
            (animation_bins.len() == 1)
                .then(|| animation_bins.values().next())
                .flatten()
        });
        let Some(animation_bin) = animation_bin else {
            println!("  Skipping animation {name}: binary not available");
            return Ok(0);
        };
        return export_animation(
            animation,
            animation_bin,
            node_by_name,
            morph_bindings,
            gltf,
            bin,
        );
    }
    let mut count = 0;
    match value {
        Value::Object(object) => {
            for child in object.values() {
                count += export_animations_from_scene(
                    child,
                    animation_bins,
                    node_by_name,
                    morph_bindings,
                    gltf,
                    bin,
                )?;
            }
        }
        Value::Array(array) => {
            for child in array {
                count += export_animations_from_scene(
                    child,
                    animation_bins,
                    node_by_name,
                    morph_bindings,
                    gltf,
                    bin,
                )?;
            }
        }
        _ => {}
    }
    Ok(count)
}

fn is_black_line_material(name: &str, material: Option<&MaterialEntry>) -> bool {
    let lower_name = name.to_ascii_lowercase();
    let material_leaf = lower_name
        .rsplit([':', '/', '\\'])
        .next()
        .unwrap_or(&lower_name);
    material_leaf.starts_with("line")
        && material.is_some_and(|material| {
            material.base_color_texture.is_none()
                && material.emissive_texture.is_none()
                && material.base_color[..3].iter().all(|value| *value <= 0.01)
        })
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn configure_vertex_colors(geometry: &mut Geometry, enabled: bool, use_alpha: bool, srgb: bool) {
    let Some(source) = geometry.attributes.remove("_SKETCHFAB_COLOR_0") else {
        return;
    };
    if !enabled || source.item_size < 3 {
        return;
    }
    let divisor = if source.normalized {
        match source.component_type {
            5121 => 255.0,
            5123 => 65535.0,
            _ => 1.0,
        }
    } else {
        1.0
    };
    let output_size = if use_alpha && source.item_size >= 4 {
        4
    } else {
        3
    };
    let source_values = source.data.to_f32_vec();
    let mut values = Vec::with_capacity(source.count * output_size);
    for color in source_values
        .chunks_exact(source.item_size)
        .take(source.count)
    {
        for value in color.iter().take(3) {
            let value = *value / divisor;
            values.push(if srgb { srgb_to_linear(value) } else { value });
        }
        if output_size == 4 {
            values.push(color[3] / divisor);
        }
    }
    geometry.attributes.insert(
        "COLOR_0".to_owned(),
        Attribute {
            data: AttrData::F32(values),
            item_size: output_size,
            count: source.count,
            component_type: 5126,
            normalized: false,
        },
    );
}

fn convert_to_glb(
    osgjs: &Value,
    poly_bin: &[u8],
    wire_bin: Option<&[u8]>,
    texture_files: &HashMap<String, TextureEntry>,
    source_materials: &HashMap<String, MaterialEntry>,
    animation_bins: &HashMap<String, Vec<u8>>,
    work_dir: &Path,
    vertex_colors: bool,
    vertex_color_alpha: bool,
    vertex_color_srgb: bool,
    flip_uvs: bool,
) -> Result<Vec<u8>> {
    println!("[5/6] Converting to glTF...");
    let flip_uvs = flip_uvs || contains_texture_attributes(osgjs);
    let mut geometries = collect_geometries(osgjs, poly_bin, wire_bin)?;
    let animated_geometry_count = geometries
        .iter()
        .filter(|geometry| geometry.animation_target.is_some())
        .count();
    if animated_geometry_count > 0 {
        println!("  {animated_geometry_count} animated geometry nodes found");
    }
    for geometry in &mut geometries {
        configure_vertex_colors(
            geometry,
            vertex_colors,
            vertex_color_alpha,
            vertex_color_srgb,
        );
        if !flip_uvs {
            for (name, attribute) in &mut geometry.attributes {
                if !name.starts_with("TEXCOORD_") {
                    continue;
                }
                if let AttrData::F32(values) = &mut attribute.data {
                    for uv in values.chunks_exact_mut(attribute.item_size) {
                        uv[1] = 1.0 - uv[1];
                    }
                }
            }
        }
    }
    geometries.retain(|geometry| {
        if is_environment_geometry(geometry, source_materials) {
            return false;
        }
        let Some(name) = geometry.material_name.as_ref() else {
            return true;
        };
        let lower_name = name.to_ascii_lowercase();
        if lower_name.contains("outline") {
            return false;
        }
        !is_black_line_material(name, source_materials.get(name))
    });
    println!("  {} geometries found", geometries.len());
    let resolved = resolved_scene(osgjs);
    let mut nodes = Vec::new();
    let mut scene_roots = Vec::new();
    let mut node_by_name = HashMap::new();
    let mut inverse_bind_by_name = HashMap::new();
    collect_skeleton_nodes(
        &resolved,
        &scene_coordinate_matrix(&resolved),
        &mut nodes,
        &mut scene_roots,
        &mut node_by_name,
        &mut inverse_bind_by_name,
        &mut HashSet::new(),
    );
    let mut gltf = json!({
        "asset": { "version": "2.0", "generator": "sketchfab-downloader-rust" },
        "scene": 0,
        "scenes": [{ "nodes": [] }],
        "nodes": [],
        "meshes": [],
        "skins": [],
        "animations": [],
        "accessors": [],
        "bufferViews": [],
        "buffers": [],
        "materials": [],
        "textures": [],
        "images": [],
        "samplers": []
    });
    let mut bin = Vec::new();
    let tex_dir = work_dir.join("textures");
    let mut image_indices = HashMap::new();
    let mut texture_uids = texture_files.keys().cloned().collect::<Vec<_>>();
    texture_uids.sort();
    for uid in texture_uids {
        let tex = &texture_files[&uid];
        if let Some(index) = add_image(&mut gltf, &mut bin, &texture_file(tex, &tex_dir))? {
            image_indices.insert(uid, index);
        }
    }
    let mut sampler_indices = HashMap::new();
    let mut texture_indices = HashMap::new();
    for material in source_materials.values() {
        for usage in [
            material.base_color_texture.as_ref(),
            material.emissive_texture.as_ref(),
            material.occlusion_texture.as_ref(),
            material.normal_texture.as_ref(),
            material.metallic_texture.as_ref(),
            material.roughness_texture.as_ref(),
            material.opacity_texture.as_ref(),
            material.alpha_mask_texture.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let key = (usage.uid.clone(), usage.sampler);
            if texture_indices.contains_key(&key) {
                continue;
            }
            let Some(image) = image_indices.get(&usage.uid).copied() else {
                continue;
            };
            let sampler = *sampler_indices
                .entry(usage.sampler)
                .or_insert_with(|| add_sampler(&mut gltf, usage.sampler));
            texture_indices.insert(key, add_texture(&mut gltf, image, sampler));
        }
    }
    let fallback_material = gltf["materials"].as_array().unwrap().len();
    gltf["materials"].as_array_mut().unwrap().push(json!({
        "name": "fallback",
        "doubleSided": true,
        "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1],
            "metallicFactor": 0,
            "roughnessFactor": 1
        }
    }));

    let mut material_indices: HashMap<(String, Vec<(u32, u32)>), usize> = HashMap::new();
    let mut metallic_roughness_indices = HashMap::new();
    let mut alpha_texture_indices = HashMap::new();
    let mut normal_texture_indices = HashMap::new();
    let mut uses_texture_transform = false;
    let mut uses_unlit = false;
    let mut used_material_extensions = HashSet::new();
    let mut morph_bindings: HashMap<String, Vec<MorphTargetBinding>> = HashMap::new();
    for geom in geometries {
        let material_index = if let Some(name) = geom.material_name.as_ref() {
            if let Some(source) = source_materials.get(name) {
                let mut units = geom
                    .texcoord_units
                    .iter()
                    .map(|(source, output)| (*source, *output))
                    .collect::<Vec<_>>();
                units.sort_unstable();
                let key = (name.clone(), units);
                if let Some(index) = material_indices.get(&key) {
                    *index
                } else {
                    let index = add_material_variant(
                        source,
                        &geom.texcoord_units,
                        &texture_indices,
                        texture_files,
                        &tex_dir,
                        &mut sampler_indices,
                        &mut metallic_roughness_indices,
                        &mut alpha_texture_indices,
                        &mut normal_texture_indices,
                        &mut uses_texture_transform,
                        flip_uvs,
                        &mut uses_unlit,
                        &mut used_material_extensions,
                        &mut gltf,
                        &mut bin,
                    )?;
                    material_indices.insert(key, index);
                    index
                }
            } else {
                fallback_material
            }
        } else {
            fallback_material
        };
        let idx_attr = Attribute {
            data: AttrData::U32(geom.indices.clone()),
            item_size: 1,
            count: geom.indices.len(),
            component_type: 5125,
            normalized: false,
        };
        let indices = add_accessor(&mut gltf, &mut bin, &idx_attr)?;
        set_accessor_target(&mut gltf, indices, 34963);
        let mut attrs = Map::new();
        for (name, attr) in &geom.attributes {
            let accessor = add_accessor(&mut gltf, &mut bin, attr)?;
            set_accessor_target(&mut gltf, accessor, 34962);
            attrs.insert(name.clone(), json!(accessor));
        }
        let mut targets = Vec::new();
        for target in &geom.morph_targets {
            let mut target_attributes = Map::new();
            for (name, attribute) in &target.attributes {
                let accessor = add_accessor(&mut gltf, &mut bin, attribute)?;
                set_accessor_target(&mut gltf, accessor, 34962);
                target_attributes.insert(name.clone(), json!(accessor));
            }
            targets.push(Value::Object(target_attributes));
        }
        let mesh_index = gltf["meshes"].as_array().unwrap().len();
        let mut primitive = json!({
            "attributes": attrs,
            "indices": indices,
            "material": material_index,
            "mode": geom.mode
        });
        if !targets.is_empty() {
            primitive["targets"] = json!(targets);
        }
        let mut mesh = json!({
            "name": geom
                .material_name
                .as_deref()
                .unwrap_or("mesh"),
            "primitives": [primitive]
        });
        if !geom.morph_targets.is_empty() {
            mesh["weights"] = json!(vec![0.0; geom.morph_targets.len()]);
            mesh["extras"] = json!({
                "targetNames": geom
                    .morph_targets
                    .iter()
                    .map(|target| target.name.clone())
                    .collect::<Vec<_>>()
            });
        }
        gltf["meshes"].as_array_mut().unwrap().push(mesh);
        let mut mesh_node = json!({
            "name": format!("mesh_{mesh_index}"),
            "mesh": mesh_index
        });
        if !geom.joint_names.is_empty() {
            let joints = geom
                .joint_names
                .iter()
                .filter_map(|name| node_by_name.get(name).copied())
                .collect::<Vec<_>>();
            if joints.len() == geom.joint_names.len() {
                let identity = matrix16(None);
                let skeleton_to_mesh = geom
                    .skeleton_matrix
                    .as_ref()
                    .and_then(invert_affine_matrix)
                    .map(|inverse| multiply_matrices(&inverse, &geom.matrix))
                    .unwrap_or(identity);
                let inverse_bind_matrices = geom
                    .joint_names
                    .iter()
                    .flat_map(|name| {
                        let source = inverse_bind_by_name.get(name).copied().unwrap_or(identity);
                        multiply_matrices(&source, &skeleton_to_mesh).into_iter()
                    })
                    .collect::<Vec<_>>();
                let inverse_bind_accessor =
                    add_mat4_accessor(&mut gltf, &mut bin, &inverse_bind_matrices);
                let skin_index = gltf["skins"].as_array().unwrap().len();
                gltf["skins"].as_array_mut().unwrap().push(json!({
                    "name": format!("skin_{mesh_index}"),
                    "joints": joints,
                    "inverseBindMatrices": inverse_bind_accessor
                }));
                mesh_node["skin"] = json!(skin_index);
            }
        }
        let node_index = nodes.len();
        for (target_index, target) in geom.morph_targets.iter().enumerate() {
            morph_bindings
                .entry(target.name.clone())
                .or_default()
                .push(MorphTargetBinding {
                    node: node_index,
                    target_index,
                    target_count: geom.morph_targets.len(),
                });
        }
        let is_skinned = mesh_node.get("skin").is_some();
        if !is_skinned {
            if let Some(target) = &geom.animation_target {
                let (translation, rotation, scale) = decompose_matrix(&geom.matrix);
                mesh_node["translation"] = json!(translation);
                mesh_node["rotation"] = json!(rotation);
                mesh_node["scale"] = json!(scale);
                node_by_name.insert(target.clone(), node_index);
            } else {
                mesh_node["matrix"] = json!(geom.matrix);
            }
        }
        nodes.push(mesh_node);
        scene_roots.push(node_index);
    }
    gltf["nodes"] = json!(nodes);
    gltf["scenes"][0]["nodes"] = json!(scene_roots);
    let animation_channel_count = export_animations_from_scene(
        &resolved,
        animation_bins,
        &node_by_name,
        &morph_bindings,
        &mut gltf,
        &mut bin,
    )?;
    if animation_channel_count == 0 {
        gltf.as_object_mut().unwrap().remove("animations");
    }
    if gltf["skins"].as_array().is_some_and(Vec::is_empty) {
        gltf.as_object_mut().unwrap().remove("skins");
    }
    let mut extensions = used_material_extensions.into_iter().collect::<Vec<_>>();
    if uses_texture_transform {
        extensions.push("KHR_texture_transform".to_owned());
    }
    if uses_unlit {
        extensions.push("KHR_materials_unlit".to_owned());
    }
    extensions.sort();
    extensions.dedup();
    if !extensions.is_empty() {
        gltf["extensionsUsed"] = json!(extensions);
    }
    if gltf["samplers"].as_array().is_some_and(Vec::is_empty) {
        gltf.as_object_mut().unwrap().remove("samplers");
    }
    if gltf["textures"].as_array().is_some_and(Vec::is_empty) {
        gltf.as_object_mut().unwrap().remove("textures");
    }
    if gltf["images"].as_array().is_some_and(Vec::is_empty) {
        gltf.as_object_mut().unwrap().remove("images");
    }
    println!("  {animation_channel_count} animation channels exported");
    gltf["buffers"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "byteLength": bin.len() }));
    write_glb(&gltf, &bin)
}

fn write_glb(gltf: &Value, bin: &[u8]) -> Result<Vec<u8>> {
    let mut json_bytes = serde_json::to_vec(gltf)?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(0x20);
    }
    let mut bin_bytes = bin.to_vec();
    while bin_bytes.len() % 4 != 0 {
        bin_bytes.push(0);
    }
    let total_len = 12 + 8 + json_bytes.len() + 8 + bin_bytes.len();
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&0x46546c67u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total_len as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4e4f534au32.to_le_bytes());
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004e4942u32.to_le_bytes());
    out.extend_from_slice(&bin_bytes);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_texture_use() -> TextureUse {
        TextureUse {
            uid: "texture".to_owned(),
            texcoord_unit: 3,
            transform: TextureTransform {
                offset: [0.0, 0.0],
                scale: [1.0, 1.0],
                rotation: 0.0,
            },
            sampler: SamplerSettings {
                mag_filter: 9729,
                min_filter: 9987,
                wrap_s: 10497,
                wrap_t: 10497,
            },
            alpha_channel: false,
        }
    }

    #[test]
    fn preserves_gltf_scene_root_coordinates() {
        let scene = json!({
            "osg.MatrixTransform": {
                "Name": "GLTF_SceneRootNode",
                "Children": []
            }
        });
        assert_eq!(scene_coordinate_matrix(&scene), matrix16(None));
    }

    #[test]
    fn converts_native_osg_coordinates() {
        let matrix = scene_coordinate_matrix(&json!({"osg.Node": {"Children": []}}));
        assert_eq!(
            matrix,
            [
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
            ]
        );
    }

    #[test]
    fn detects_scene_texture_bindings() {
        assert!(contains_texture_attributes(&json!({
            "osg.StateSet": {
                "TextureAttributeList": [[], []]
            }
        })));
        assert!(!contains_texture_attributes(
            &json!({"osg.Node": {"Children": []}})
        ));
    }

    fn black_material() -> MaterialEntry {
        MaterialEntry {
            name: "line".to_owned(),
            base_color: [0.0, 0.0, 0.0, 1.0],
            base_color_texture: None,
            emissive_color: [0.0, 0.0, 0.0],
            emissive_enabled: false,
            emissive_texture: None,
            occlusion_texture: None,
            normal_texture: None,
            metallic_texture: None,
            roughness_texture: None,
            roughness_invert: false,
            opacity_texture: None,
            alpha_mask_texture: None,
            alpha_invert: false,
            normal_scale: 1.0,
            normal_flip_y: false,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            alpha_mode: "OPAQUE",
            alpha_cutoff: 0.5,
            double_sided: true,
            unlit: false,
            extensions: Map::new(),
        }
    }

    #[test]
    fn filters_namespaced_black_line_materials() {
        let material = black_material();
        assert!(is_black_line_material("model:model:Linea", Some(&material)));
    }

    #[test]
    fn preserves_textured_line_materials() {
        let mut material = black_material();
        material.base_color_texture = Some(test_texture_use());
        assert!(!is_black_line_material("LineArt", Some(&material)));
    }

    #[test]
    fn maps_sparse_source_uv_units_to_dense_gltf_units() {
        let usage = test_texture_use();
        let indices = HashMap::from([((usage.uid.clone(), usage.sampler), 4)]);
        let units = HashMap::from([(3, 0)]);
        let mut uses_transform = false;
        let info = texture_info(&usage, &indices, &units, &mut uses_transform, true).unwrap();
        assert_eq!(info["index"], 4);
        assert_eq!(info["texCoord"], 0);
        assert!(!uses_transform);
    }

    #[test]
    fn converts_texture_transform_after_uv_flip() {
        let mut usage = test_texture_use();
        usage.transform.offset = [0.1, 0.2];
        usage.transform.scale = [2.0, 3.0];
        let mut uses_transform = false;
        let info = texture_info_for_index(
            0,
            &usage,
            &HashMap::from([(3, 0)]),
            &mut uses_transform,
            true,
        );
        assert!(uses_transform);
        let offset = info
            .pointer("/extensions/KHR_texture_transform/offset")
            .and_then(Value::as_array)
            .unwrap();
        assert!((offset[0].as_f64().unwrap() - 0.1).abs() < 1e-6);
        assert!((offset[1].as_f64().unwrap() + 2.2).abs() < 1e-6);
    }

    #[test]
    fn samples_morph_tracks_linearly() {
        let track = ScalarTrack {
            times: vec![0.0, 2.0],
            values: vec![0.0, 1.0],
        };
        assert_eq!(sample_scalar_track(&track, -1.0), 0.0);
        assert_eq!(sample_scalar_track(&track, 1.0), 0.5);
        assert_eq!(sample_scalar_track(&track, 3.0), 1.0);
    }

    #[test]
    fn deduplicates_animation_times_using_last_value() {
        let (times, values) =
            deduplicate_keyframes(vec![0.0, 1.0, 1.0, 2.0], vec![0.0, 1.0, 2.0, 3.0], 1);
        assert_eq!(times, vec![0.0, 1.0, 2.0]);
        assert_eq!(values, vec![0.0, 2.0, 3.0]);
    }
}
