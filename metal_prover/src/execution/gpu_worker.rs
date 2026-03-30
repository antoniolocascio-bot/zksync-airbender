///! Single GPU worker thread for Metal.
///!
///! Unlike the CUDA version which manages multiple GPU workers across
///! different devices, the Metal version has a single worker since
///! Apple Silicon has one GPU.

use std::sync::mpsc;
use std::thread;

use crate::circuit_type::CircuitType;
use crate::prover::context::{MetalProverContext, MetalProverContextConfig};
use crate::prover::setup::SetupPrecomputations;

/// Messages sent to the GPU worker thread.
pub enum GpuWorkerMessage {
    /// Submit a proof generation job.
    Prove {
        circuit_type: CircuitType,
    },
    /// Shut down the worker.
    Shutdown,
}

/// Messages sent back from the GPU worker thread.
pub enum GpuWorkerResponse {
    /// A proof job completed.
    ProofComplete,
    /// The worker has shut down.
    ShutdownComplete,
    /// An error occurred.
    Error(String),
}

/// A GPU worker that processes proof jobs on a dedicated thread.
///
/// The worker owns the MetalProverContext and processes jobs sequentially.
/// This ensures that Metal command buffers are submitted from a consistent
/// thread (recommended by Metal best practices).
pub struct GpuWorker {
    sender: mpsc::Sender<GpuWorkerMessage>,
    receiver: mpsc::Receiver<GpuWorkerResponse>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl GpuWorker {
    /// Spawn a new GPU worker thread.
    pub fn new(config: MetalProverContextConfig) -> Self {
        let (msg_tx, msg_rx) = mpsc::channel::<GpuWorkerMessage>();
        let (resp_tx, resp_rx) = mpsc::channel::<GpuWorkerResponse>();

        let thread_handle = thread::spawn(move || {
            let mut ctx = MetalProverContext::new(config);
            log::info!(
                "GPU worker started on device: {}",
                ctx.device_name()
            );

            loop {
                match msg_rx.recv() {
                    Ok(GpuWorkerMessage::Prove { circuit_type }) => {
                        ctx.reset_allocator();
                        // TODO: Accept setup precomputations through the message.
                        // For now, just signal completion.
                        resp_tx.send(GpuWorkerResponse::ProofComplete).ok();
                    }
                    Ok(GpuWorkerMessage::Shutdown) => {
                        log::info!("GPU worker shutting down");
                        resp_tx.send(GpuWorkerResponse::ShutdownComplete).ok();
                        break;
                    }
                    Err(_) => {
                        // Channel disconnected, exit
                        break;
                    }
                }
            }
        });

        Self {
            sender: msg_tx,
            receiver: resp_rx,
            thread_handle: Some(thread_handle),
        }
    }

    /// Submit a proof job to the worker.
    pub fn submit_proof(&self, circuit_type: CircuitType) {
        self.sender
            .send(GpuWorkerMessage::Prove { circuit_type })
            .expect("GPU worker channel disconnected");
    }

    /// Wait for the next response from the worker.
    pub fn wait_response(&self) -> GpuWorkerResponse {
        self.receiver
            .recv()
            .unwrap_or(GpuWorkerResponse::Error("Channel disconnected".to_string()))
    }

    /// Shut down the worker and wait for the thread to exit.
    pub fn shutdown(mut self) {
        self.sender.send(GpuWorkerMessage::Shutdown).ok();
        if let Some(handle) = self.thread_handle.take() {
            handle.join().ok();
        }
    }
}

impl Drop for GpuWorker {
    fn drop(&mut self) {
        self.sender.send(GpuWorkerMessage::Shutdown).ok();
        if let Some(handle) = self.thread_handle.take() {
            handle.join().ok();
        }
    }
}
