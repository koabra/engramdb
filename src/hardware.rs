//! Honest GPU/GDS capability detection and optional libcufile bindings.

use std::env;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferPath {
    CpuDirect,
    GdsCompatMode,
    GdsDirectVerified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    pub linux: bool,
    pub nvidia_driver: bool,
    pub nvidia_fs: bool,
    pub cuda_visible: bool,
    pub libcufile_path: Option<PathBuf>,
    pub cufile_driver_opened: bool,
    pub gdsio_path: Option<PathBuf>,
    pub transfer_path: TransferPath,
    pub notes: Vec<String>,
}

impl HardwareCapabilities {
    pub fn detect() -> Self {
        let linux = cfg!(target_os = "linux");
        let nvidia_driver = Path::new("/proc/driver/nvidia/version").exists();
        let nvidia_fs = Path::new("/dev/nvidia-fs").exists()
            || Path::new("/dev/nvidia-fs0").exists()
            || Path::new("/proc/driver/nvidia-fs").exists();
        let cuda_visible = env::var("CUDA_VISIBLE_DEVICES")
            .map(|value| !value.trim().is_empty() && value.trim() != "-1")
            .unwrap_or(nvidia_driver);
        let gdsio_path = executable_on_path("gdsio");
        let mut notes = Vec::new();
        #[cfg(feature = "gds")]
        let (libcufile_path, cufile_driver_opened) = match GdsApi::probe() {
            Ok((path, opened)) => {
                notes.push(
                    "libcufile symbols loaded; direct P2P was not verified by I/O stats".to_owned(),
                );
                (Some(path), opened)
            }
            Err(error) => {
                notes.push(error.to_string());
                (None, false)
            }
        };
        #[cfg(not(feature = "gds"))]
        let (libcufile_path, cufile_driver_opened) = {
            notes.push("binary compiled without the optional `gds` feature".to_owned());
            (None, false)
        };

        let transfer_path =
            if libcufile_path.is_some() && cufile_driver_opened && nvidia_driver && nvidia_fs {
                notes.push(
                    "classified conservatively as compat/unverified until P2P counters are observed"
                        .to_owned(),
                );
                TransferPath::GdsCompatMode
            } else {
                notes.push("using 4 KiB-aligned CPU direct-I/O fallback".to_owned());
                TransferPath::CpuDirect
            };
        Self {
            linux,
            nvidia_driver,
            nvidia_fs,
            cuda_visible,
            libcufile_path,
            cufile_driver_opened,
            gdsio_path,
            transfer_path,
            notes,
        }
    }

    pub fn supports_verified_gds(&self) -> bool {
        self.transfer_path == TransferPath::GdsDirectVerified
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|error| Error::Arrow(error.to_string()))
    }
}

fn executable_on_path(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .map(|path| path.join(name))
            .find(|candidate| candidate.is_file())
    })
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CufileError {
    pub error_code: i32,
    pub cuda_error: i32,
}

impl CufileError {
    pub fn is_success(self) -> bool {
        self.error_code == 0 && self.cuda_error == 0
    }
}

pub type CufileHandle = *mut libc::c_void;
pub type CudaStream = *mut libc::c_void;

#[cfg(feature = "gds")]
type DriverOpenFn = unsafe extern "C" fn() -> CufileError;
#[cfg(feature = "gds")]
type DriverCloseFn = unsafe extern "C" fn() -> CufileError;
#[cfg(feature = "gds")]
type ReadFn = unsafe extern "C" fn(CufileHandle, *mut libc::c_void, usize, i64, i64) -> isize;
#[cfg(feature = "gds")]
type WriteFn = unsafe extern "C" fn(CufileHandle, *const libc::c_void, usize, i64, i64) -> isize;
#[cfg(feature = "gds")]
type ReadAsyncFn = unsafe extern "C" fn(
    CufileHandle,
    *mut libc::c_void,
    *mut usize,
    *mut i64,
    *mut i64,
    *mut isize,
    CudaStream,
) -> CufileError;
#[cfg(feature = "gds")]
type WriteAsyncFn = unsafe extern "C" fn(
    CufileHandle,
    *const libc::c_void,
    *mut usize,
    *mut i64,
    *mut i64,
    *mut isize,
    CudaStream,
) -> CufileError;

#[cfg(feature = "gds")]
pub struct GdsApi {
    _library: libloading::Library,
    driver_close: DriverCloseFn,
    read: ReadFn,
    write: WriteFn,
    read_async: ReadAsyncFn,
    write_async: WriteAsyncFn,
}

#[cfg(feature = "gds")]
impl GdsApi {
    pub fn open() -> Result<(Self, PathBuf)> {
        let candidates = libcufile_candidates();
        let mut errors = Vec::new();
        for candidate in candidates {
            // SAFETY: the library stays owned by GdsApi for all copied symbols.
            let library = match unsafe { libloading::Library::new(&candidate) } {
                Ok(library) => library,
                Err(error) => {
                    errors.push(format!("{}: {error}", candidate.display()));
                    continue;
                }
            };
            // SAFETY: symbol signatures mirror the public libcufile C API.
            let driver_open = unsafe {
                *library
                    .get::<DriverOpenFn>(b"cuFileDriverOpen\0")
                    .map_err(|error| Error::HardwareUnavailable(error.to_string()))?
            };
            // SAFETY: see driver_open.
            let driver_close = unsafe {
                *library
                    .get::<DriverCloseFn>(b"cuFileDriverClose\0")
                    .map_err(|error| Error::HardwareUnavailable(error.to_string()))?
            };
            // SAFETY: see driver_open.
            let read = unsafe {
                *library
                    .get::<ReadFn>(b"cuFileRead\0")
                    .map_err(|error| Error::HardwareUnavailable(error.to_string()))?
            };
            // SAFETY: see driver_open.
            let write = unsafe {
                *library
                    .get::<WriteFn>(b"cuFileWrite\0")
                    .map_err(|error| Error::HardwareUnavailable(error.to_string()))?
            };
            // SAFETY: see driver_open.
            let read_async = unsafe {
                *library
                    .get::<ReadAsyncFn>(b"cuFileReadAsync\0")
                    .map_err(|error| Error::HardwareUnavailable(error.to_string()))?
            };
            // SAFETY: see driver_open.
            let write_async = unsafe {
                *library
                    .get::<WriteAsyncFn>(b"cuFileWriteAsync\0")
                    .map_err(|error| Error::HardwareUnavailable(error.to_string()))?
            };
            // SAFETY: libcufile owns process-global driver state.
            let opened = unsafe { driver_open() };
            if !opened.is_success() {
                return Err(Error::HardwareUnavailable(format!(
                    "cuFileDriverOpen failed: {} / {}",
                    opened.error_code, opened.cuda_error
                )));
            }
            return Ok((
                Self {
                    _library: library,
                    driver_close,
                    read,
                    write,
                    read_async,
                    write_async,
                },
                candidate,
            ));
        }
        Err(Error::HardwareUnavailable(format!(
            "libcufile could not be loaded ({})",
            errors.join("; ")
        )))
    }

    fn probe() -> Result<(PathBuf, bool)> {
        let (api, path) = Self::open()?;
        drop(api);
        Ok((path, true))
    }

    /// # Safety
    ///
    /// `handle` must be registered with libcufile and `device_pointer` must
    /// address at least `size` writable bytes in the active CUDA context.
    pub unsafe fn read_sync(
        &self,
        handle: CufileHandle,
        device_pointer: *mut libc::c_void,
        size: usize,
        file_offset: i64,
        buffer_offset: i64,
    ) -> Result<usize> {
        // SAFETY: guaranteed by caller.
        let result =
            unsafe { (self.read)(handle, device_pointer, size, file_offset, buffer_offset) };
        if result < 0 {
            return Err(Error::HardwareUnavailable(format!(
                "cuFileRead failed with {result}"
            )));
        }
        Ok(result as usize)
    }

    /// # Safety
    ///
    /// `handle`, `device_pointer`, and `stream` must be valid for libcufile and
    /// all in/out pointers must remain alive until CUDA stream completion.
    pub unsafe fn read_async(
        &self,
        handle: CufileHandle,
        device_pointer: *mut libc::c_void,
        size: &mut usize,
        file_offset: &mut i64,
        buffer_offset: &mut i64,
        bytes_read: &mut isize,
        stream: CudaStream,
    ) -> Result<()> {
        // SAFETY: guaranteed by caller.
        let error = unsafe {
            (self.read_async)(
                handle,
                device_pointer,
                size,
                file_offset,
                buffer_offset,
                bytes_read,
                stream,
            )
        };
        map_cufile_error("cuFileReadAsync", error)
    }

    /// # Safety
    ///
    /// Same requirements as [`Self::read_sync`], except the device memory must
    /// be readable by libcufile.
    pub unsafe fn write_sync(
        &self,
        handle: CufileHandle,
        device_pointer: *const libc::c_void,
        size: usize,
        file_offset: i64,
        buffer_offset: i64,
    ) -> Result<usize> {
        // SAFETY: guaranteed by caller.
        let result =
            unsafe { (self.write)(handle, device_pointer, size, file_offset, buffer_offset) };
        if result < 0 {
            return Err(Error::HardwareUnavailable(format!(
                "cuFileWrite failed with {result}"
            )));
        }
        Ok(result as usize)
    }

    /// # Safety
    ///
    /// Same requirements as [`Self::read_async`], except the device memory
    /// must be readable by libcufile.
    pub unsafe fn write_async(
        &self,
        handle: CufileHandle,
        device_pointer: *const libc::c_void,
        size: &mut usize,
        file_offset: &mut i64,
        buffer_offset: &mut i64,
        bytes_written: &mut isize,
        stream: CudaStream,
    ) -> Result<()> {
        // SAFETY: guaranteed by caller.
        let error = unsafe {
            (self.write_async)(
                handle,
                device_pointer,
                size,
                file_offset,
                buffer_offset,
                bytes_written,
                stream,
            )
        };
        map_cufile_error("cuFileWriteAsync", error)
    }
}

#[cfg(feature = "gds")]
impl Drop for GdsApi {
    fn drop(&mut self) {
        // SAFETY: this instance successfully opened the process-global driver.
        let _ = unsafe { (self.driver_close)() };
    }
}

#[cfg(feature = "gds")]
fn map_cufile_error(operation: &str, error: CufileError) -> Result<()> {
    if error.is_success() {
        Ok(())
    } else {
        Err(Error::HardwareUnavailable(format!(
            "{operation} failed: {} / {}",
            error.error_code, error.cuda_error
        )))
    }
}

#[cfg(feature = "gds")]
fn libcufile_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("CUFILE_PATH") {
        candidates.push(PathBuf::from(path));
    }
    candidates.extend(
        [
            "/usr/local/cuda/gds/lib64/libcufile.so",
            "/usr/local/cuda/lib64/libcufile.so",
            "libcufile.so.0",
            "libcufile.so",
        ]
        .into_iter()
        .map(PathBuf::from),
    );
    candidates
}
