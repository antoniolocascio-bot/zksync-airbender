///! MetalProverContext: the main context holding device, command queue,
///! compiled shader library, pipeline cache, device context, and allocator.

use metal::{CommandQueue, ComputePipelineState, Device as MTLDevice, Library};
use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use crate::allocator::SharedAllocator;
use crate::device_context::DeviceContext;

/// Configuration for the Metal prover context.
#[derive(Copy, Clone, Debug)]
pub struct MetalProverContextConfig {
    /// Log2 of the coarsest twiddle factor layer count.
    pub powers_of_w_coarse_log_count: u32,
    /// Log2 of the allocator block size in bytes.
    pub allocator_block_log_size: u32,
    /// Number of pre-allocated allocator blocks.
    pub allocator_num_blocks: usize,
}

impl Default for MetalProverContextConfig {
    fn default() -> Self {
        Self {
            powers_of_w_coarse_log_count: 12,
            allocator_block_log_size: 20,  // 1 MB blocks
            allocator_num_blocks: 256,     // 256 MB total
        }
    }
}

/// Approximate device properties for Metal GPUs.
/// Unlike CUDA, Metal does not expose SM count or L2 cache size directly,
/// but we provide estimated values for tuning.
pub struct DeviceProperties {
    /// Estimated GPU execution units (used for occupancy heuristics).
    pub gpu_core_count: usize,
    /// Maximum threadgroup memory in bytes.
    pub max_threadgroup_memory: usize,
    /// Maximum threads per threadgroup.
    pub max_threads_per_threadgroup: usize,
}

impl DeviceProperties {
    pub fn new(device: &MTLDevice) -> Self {
        Self {
            // Apple Silicon typically has 8-128 GPU cores depending on chip
            gpu_core_count: 32, // conservative default
            max_threadgroup_memory: device.max_threadgroup_memory_length() as usize,
            max_threads_per_threadgroup: device.max_threads_per_threadgroup().width as usize,
        }
    }
}

/// Cache for compiled compute pipeline states, keyed by kernel function name.
pub struct PipelineCache {
    library: Library,
    cache: RwLock<HashMap<String, ComputePipelineState>>,
}

impl PipelineCache {
    pub fn new(library: Library) -> Self {
        Self {
            library,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Resolve a kernel name to its fully-qualified name in the Metal library.
    ///
    /// Metal shaders are organized in namespaces (e.g. `airbender::ops_simple::`).
    /// This resolves a short name like `"ab_set_by_val_bf_kernel"` to the full
    /// qualified name by trying known namespace prefixes.
    fn resolve_kernel_name(&self, name: &str) -> String {
        // If the name already contains `::` it is fully qualified -- use as-is.
        if name.contains("::") {
            return name.to_string();
        }
        // Try each known namespace prefix in order.
        const NAMESPACES: &[&str] = &[
            "airbender::ops_simple::",
            "airbender::ops_complex::",
            "airbender::ops_cub::",
            "airbender::blake2s::",
            "airbender::monolith::",
            "airbender::ntt::",
            "airbender::stage2::",
            "airbender::stage3::",
            "airbender::stage4::",
        ];
        for ns in NAMESPACES {
            let qualified = format!("{}{}", ns, name);
            if self.library.get_function(&qualified, None).is_ok() {
                return qualified;
            }
        }
        // Fall back to bare name; get_function will panic with a useful message.
        name.to_string()
    }

    /// Get or create a compute pipeline state for the named kernel function.
    ///
    /// Kernel names may be given with or without namespace prefix; this method
    /// resolves them automatically by trying known `airbender::*` namespaces.
    ///
    /// Thread-safe: uses a RwLock so multiple readers can access cached
    /// pipelines concurrently.
    pub fn get_or_create(&self, name: &str, device: &MTLDevice) -> ComputePipelineState {
        // Fast path: check if already cached (use the short name as key)
        {
            let cache = self.cache.read().unwrap();
            if let Some(pipeline) = cache.get(name) {
                return pipeline.clone();
            }
        }

        // Resolve the kernel name (handles bare vs. namespaced names)
        let qualified_name = self.resolve_kernel_name(name);

        // Slow path: compile and insert
        let function = self
            .library
            .get_function(&qualified_name, None)
            .unwrap_or_else(|e| panic!("Failed to find Metal function '{}' (resolved from '{}'): {:?}", qualified_name, name, e));
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to create compute pipeline for '{}': {:?}",
                    qualified_name, e
                )
            });

        let mut cache = self.cache.write().unwrap();
        cache.insert(name.to_string(), pipeline.clone());
        pipeline
    }
}

/// The main Metal prover context.
///
/// Holds the Metal device, command queue, compiled shader library,
/// pipeline cache, precomputed twiddle factors, and memory allocator.
///
/// Unlike the CUDA `ProverContext` which manages multiple streams for
/// overlapping transfers, the Metal context uses a single command queue
/// (Apple Silicon's unified memory eliminates the need for H2D copies).
pub struct MetalProverContext {
    pub device: MTLDevice,
    pub command_queue: CommandQueue,
    pub pipeline_cache: PipelineCache,
    pub device_context: DeviceContext,
    pub allocator: SharedAllocator,
    pub device_properties: DeviceProperties,
}

impl MetalProverContext {
    /// Create a new Metal prover context with default configuration.
    pub fn new(config: MetalProverContextConfig) -> Self {
        Self::new_with_metallib(config, None)
    }

    /// Create a new Metal prover context, optionally loading a specific .metallib file.
    ///
    /// If `metallib_path` is None, the default library is loaded from the device.
    pub fn new_with_metallib(
        config: MetalProverContextConfig,
        metallib_path: Option<&Path>,
    ) -> Self {
        let device = MTLDevice::system_default()
            .expect("No Metal-capable GPU found. Metal prover requires Apple Silicon.");

        let command_queue = device.new_command_queue();

        let library = match metallib_path {
            Some(path) => device
                .new_library_with_file(path)
                .unwrap_or_else(|e| panic!("Failed to load metallib from {:?}: {:?}", path, e)),
            None => device.new_default_library(),
        };

        let pipeline_cache = PipelineCache::new(library);

        let device_context = DeviceContext::create(&device, config.powers_of_w_coarse_log_count);

        let block_size = 1usize << config.allocator_block_log_size;
        let allocator = SharedAllocator::new(&device, block_size, config.allocator_num_blocks);

        let device_properties = DeviceProperties::new(&device);

        Self {
            device,
            command_queue,
            pipeline_cache,
            device_context,
            allocator,
            device_properties,
        }
    }

    /// Get a compute pipeline state for a named kernel function.
    /// Pipelines are cached after first creation.
    pub fn get_pipeline(&self, name: &str) -> ComputePipelineState {
        self.pipeline_cache.get_or_create(name, &self.device)
    }

    /// Reset the arena allocator, freeing all temporary allocations.
    pub fn reset_allocator(&mut self) {
        self.allocator.reset();
    }

    /// Get the Metal device name (e.g., "Apple M1 Max").
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }
}
