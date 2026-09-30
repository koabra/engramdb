//! A small hazard-pointer protected atomic `Arc`.
//!
//! Readers publish the raw pointer before validating it and incrementing the
//! strong count. Replaced pointers are retired and reclaimed only after no
//! hazard slot references them. This keeps the read path lock-free while
//! retaining ergonomic `Arc<T>` ownership after protection.

use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr;
use std::sync::atomic::{fence, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex, Weak};

struct Slot<T> {
    pointer: AtomicPtr<T>,
}

impl<T> Slot<T> {
    fn new() -> Self {
        Self {
            pointer: AtomicPtr::new(ptr::null_mut()),
        }
    }
}

struct Retired<T>(*const T);

// A retired pointer originated from Arc::into_raw. Access and destruction are
// serialized by DomainInner::retired, and T is required to be thread-safe.
unsafe impl<T: Send + Sync> Send for Retired<T> {}

struct DomainInner<T> {
    slots: Mutex<Vec<Weak<Slot<T>>>>,
    retired: Mutex<Vec<Retired<T>>>,
}

impl<T> Drop for DomainInner<T> {
    fn drop(&mut self) {
        let retired = self.retired.get_mut().expect("retired mutex poisoned");
        for pointer in retired.drain(..) {
            // SAFETY: every retired pointer represents exactly one strong count
            // transferred by Arc::into_raw and is drained exactly once.
            unsafe { drop(Arc::from_raw(pointer.0)) };
        }
    }
}

/// Registry of hazard slots and retired `Arc` pointers.
pub struct HazardDomain<T> {
    inner: Arc<DomainInner<T>>,
}

impl<T> Clone for HazardDomain<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Send + Sync + 'static> Default for HazardDomain<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync + 'static> HazardDomain<T> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DomainInner {
                slots: Mutex::new(Vec::new()),
                retired: Mutex::new(Vec::new()),
            }),
        }
    }

    fn slot(&self) -> Arc<Slot<T>> {
        let slot = Arc::new(Slot::new());
        self.inner
            .slots
            .lock()
            .expect("slot mutex poisoned")
            .push(Arc::downgrade(&slot));
        slot
    }

    fn retire(&self, pointer: *const T) {
        if pointer.is_null() {
            return;
        }
        self.inner
            .retired
            .lock()
            .expect("retired mutex poisoned")
            .push(Retired(pointer));
        self.collect();
    }

    /// Reclaim retired pointers that are not currently protected.
    pub fn collect(&self) {
        let hazards = {
            let mut slots = self.inner.slots.lock().expect("slot mutex poisoned");
            slots.retain(|slot| slot.strong_count() > 0);
            slots
                .iter()
                .filter_map(Weak::upgrade)
                .map(|slot| slot.pointer.load(Ordering::SeqCst) as *const T)
                .filter(|pointer| !pointer.is_null())
                .collect::<Vec<_>>()
        };

        let mut retired = self.inner.retired.lock().expect("retired mutex poisoned");
        let mut index = 0;
        while index < retired.len() {
            if hazards.contains(&retired[index].0) {
                index += 1;
            } else {
                let pointer = retired.swap_remove(index).0;
                // SAFETY: the pointer is no longer published by the owner, no
                // reader protects it, and its raw strong count is reclaimed once.
                unsafe { drop(Arc::from_raw(pointer)) };
            }
        }
    }

    pub fn retired_count(&self) -> usize {
        self.inner
            .retired
            .lock()
            .expect("retired mutex poisoned")
            .len()
    }
}

/// An atomic owning one raw `Arc<T>` strong reference.
pub struct HazardAtomic<T: Send + Sync + 'static> {
    pointer: AtomicPtr<T>,
    domain: HazardDomain<T>,
}

impl<T: Send + Sync + 'static> HazardAtomic<T> {
    pub fn new(value: Arc<T>, domain: HazardDomain<T>) -> Self {
        Self {
            pointer: AtomicPtr::new(Arc::into_raw(value) as *mut T),
            domain,
        }
    }

    /// Protect and clone the current value without locking.
    pub fn load(&self) -> HazardGuard<T> {
        let slot = self.domain.slot();
        loop {
            let pointer = self.pointer.load(Ordering::SeqCst);
            debug_assert!(!pointer.is_null());
            slot.pointer.store(pointer, Ordering::SeqCst);
            fence(Ordering::SeqCst);
            if self.pointer.load(Ordering::SeqCst) == pointer {
                // SAFETY: the matching hazard slot prevents reclamation between
                // validation and incrementing this strong count.
                unsafe { Arc::increment_strong_count(pointer) };
                slot.pointer.store(ptr::null_mut(), Ordering::SeqCst);
                // SAFETY: increment_strong_count created the count consumed here.
                let value = unsafe { Arc::from_raw(pointer) };
                return HazardGuard {
                    value,
                    _not_send_slot: PhantomData,
                };
            }
            slot.pointer.store(ptr::null_mut(), Ordering::SeqCst);
        }
    }

    pub fn store(&self, value: Arc<T>) {
        let new_pointer = Arc::into_raw(value) as *mut T;
        let old_pointer = self.pointer.swap(new_pointer, Ordering::SeqCst);
        self.domain.retire(old_pointer);
    }
}

impl<T: Send + Sync + 'static> Drop for HazardAtomic<T> {
    fn drop(&mut self) {
        let pointer = self.pointer.swap(ptr::null_mut(), Ordering::SeqCst);
        self.domain.retire(pointer);
    }
}

/// Owned value returned after hazard protection. It dereferences as `T`.
pub struct HazardGuard<T> {
    value: Arc<T>,
    _not_send_slot: PhantomData<*const ()>,
}

impl<T> Clone for HazardGuard<T> {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
            _not_send_slot: PhantomData,
        }
    }
}

impl<T> Deref for HazardGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> HazardGuard<T> {
    pub(crate) fn from_arc(value: Arc<T>) -> Self {
        Self {
            value,
            _not_send_slot: PhantomData,
        }
    }

    pub fn into_arc(self) -> Arc<T> {
        self.value
    }
}
