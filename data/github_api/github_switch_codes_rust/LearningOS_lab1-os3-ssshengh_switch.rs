// Repository: LearningOS/lab1-os3-ssshengh
// File: os3/src/task/switch.rs

use crate::task::task_context::TaskContext;
core::arch::global_asm!(include_str!("switch.S"));

extern "C" {
    /// Switch to the context of `next_task_cx_ptr`, saving the current context
    /// in `current_task_cx_ptr`.
    pub fn __switch(current_task_cx_ptr: *mut TaskContext, next_task_cx_ptr: *const TaskContext);
}