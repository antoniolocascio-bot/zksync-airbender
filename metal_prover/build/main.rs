use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(no_metal)");

    // Metal is only available on macOS
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if target_os != "macos" {
        println!("cargo::warning=Metal is only available on macOS. Building with no_metal cfg.");
        println!("cargo::rustc-cfg=no_metal");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let native_dir = Path::new("native");

    // Collect all .metal files
    let metal_files = collect_metal_files(native_dir);

    if metal_files.is_empty() {
        println!("cargo::warning=No .metal files found in native/");
        println!("cargo::rustc-cfg=no_metal");
        return;
    }

    // Compile each .metal file to .air
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let file_stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_file = out_dir.join(format!("{}.air", file_stem));

        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                "-frecord-sources",
                "-o",
                air_file.to_str().unwrap(),
                metal_file.to_str().unwrap(),
                // Include the native directory for header resolution
                "-I",
                native_dir.to_str().unwrap(),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                air_files.push(air_file);
            }
            Ok(s) => {
                println!(
                    "cargo::warning=Metal compilation failed for {:?} with status {}",
                    metal_file, s
                );
                println!("cargo::rustc-cfg=no_metal");
                return;
            }
            Err(e) => {
                println!(
                    "cargo::warning=Failed to run xcrun metal: {}. Is Xcode installed?",
                    e
                );
                println!("cargo::rustc-cfg=no_metal");
                return;
            }
        }

        // Track source file for rebuilds
        println!("cargo:rerun-if-changed={}", metal_file.display());
    }

    // Link all .air files into a single .metallib
    let metallib_path = out_dir.join("airbender.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args([
        "-sdk",
        "macosx",
        "metallib",
        "-o",
        metallib_path.to_str().unwrap(),
    ]);
    for air_file in &air_files {
        cmd.arg(air_file.to_str().unwrap());
    }

    let status = cmd.status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo::warning=metallib linking failed with status {}", s);
            println!("cargo::rustc-cfg=no_metal");
            return;
        }
        Err(e) => {
            println!("cargo::warning=Failed to run xcrun metallib: {}", e);
            println!("cargo::rustc-cfg=no_metal");
            return;
        }
    }

    // Export the metallib path for runtime loading
    println!(
        "cargo:rustc-env=METAL_LIB_PATH={}",
        metallib_path.display()
    );

    // Rerun if any native file changes
    println!("cargo:rerun-if-changed=native/");
}

fn collect_metal_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_metal_files(&path));
            } else if path.extension().map_or(false, |ext| ext == "metal") {
                files.push(path);
            }
        }
    }
    files
}
