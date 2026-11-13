//! Picking the device the network runs on.

use std::panic::{self, AssertUnwindSafe};

use burn::tensor::activation::{log_softmax, tanh};
use burn::tensor::{Device, DeviceType, Tensor};
use tracing::{info, warn};

use crate::util::panic_message;

// The device every command runs the network on: the first CUDA GPU if there is one and it
// works, otherwise the Flex CPU backend. Setting BURN_DEVICE (cuda, flex, ...) skips the check
// and uses that backend, as burn's own Device::default() does.
//
// Device::default() alone isn't enough: with the cuda feature compiled in it always picks
// CUDA, and a machine without an NVIDIA driver or GPU only finds out when the first tensor
// operation panics.
pub fn pick_device() -> Device {
    if let Ok(name) = std::env::var("BURN_DEVICE") {
        let device = Device::default();
        info!("Using device {:?} (BURN_DEVICE={})", device, name);
        return device;
    }
    // Safe without a driver: it checks that libcuda loads before asking it for devices.
    let Some(cuda) = Device::enumerate(DeviceType::Cuda).into_iter().next() else {
        warn!("No CUDA GPU found; running on the CPU (Flex backend), which is much slower");
        return Device::flex();
    };
    return match smoke_test(&cuda) {
        Ok(()) => {
            info!("Using device {:?}", cuda);
            cuda
        }
        Err(e) => {
            warn!(
                "CUDA GPU found but not usable, so running on the CPU (Flex backend), which is \
                 much slower. To use the GPU, install the CUDA Toolkit (and set CUDA_PATH to it \
                 if it isn't found). The error: {}",
                root_cause(&e)
            );
            Device::flex()
        }
    };
}

// Runs a tiny computation on `device`, returning the panic message if burn can't. It uses the
// math functions the network, its loss and AdamW do (exp, log via log_softmax, tanh, sqrt):
// on CUDA those compile to calls into libdevice, which comes with the CUDA Toolkit rather than
// the driver, so a machine with only the driver can run simple kernels but not these. The
// default panic hook is silenced meanwhile so an expected failure doesn't print a backtrace;
// call this before starting other threads, since the hook is process-wide.
fn smoke_test(device: &Device) -> Result<(), String> {
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let xs = [0.5f32, 1.0, 2.0];
        let t = Tensor::<1>::from_floats(xs, device);
        let y =
            t.clone().exp().log() + tanh(t.clone()) + t.clone().sqrt() + log_softmax(t, 0).exp();
        let got = y.sum().into_scalar::<f32>();
        // log_softmax(t).exp() sums to 1.
        let want = 1.0 + xs.iter().map(|x| x + x.tanh() + x.sqrt()).sum::<f32>();
        assert!(
            (got - want).abs() < 1e-3,
            "computed {} instead of {}",
            got,
            want
        );
    }));
    panic::set_hook(hook);
    return result.map_err(|p| panic_message(&p).to_string());
}

// The innermost cause of a burn/cubecl error, whose message nests each cause under a
// "Caused by:" line and ends with backtrace sections; falls back to the first line.
fn root_cause(msg: &str) -> &str {
    let lines: Vec<&str> = msg.lines().collect();
    if let Some(i) = lines.iter().rposition(|l| l.trim() == "Caused by:") {
        if let Some(cause) = lines.get(i + 1) {
            return cause.trim();
        }
    }
    return lines.first().map_or("", |l| l.trim());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_cause_is_the_innermost_cause() {
        let msg = "Error while reading data\nCaused by:\n  A launch error happened\nCaused by:\n  \
                   libdevice.10.bc was not found\nBacktrace:\nNo backtrace available";
        assert_eq!(root_cause(msg), "libdevice.10.bc was not found");
        assert_eq!(root_cause("no causes here\nsecond line"), "no causes here");
        assert_eq!(root_cause(""), "");
    }
}
