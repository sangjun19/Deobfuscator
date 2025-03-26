// Repository: HOMS-OSS/ruperf
// File: src/test/counts.rs

use std::io::Read;
use std::io::Write;

use crate::event::open::Event;
use crate::stat::StatEvent;
use crate::test::RunSettings;
use crate::test::Test;
use crate::test::TestResult;
use os_pipe::pipe;
use std::os::unix::process::CommandExt;
use std::process::Command;

pub fn launch_test_process(
    command: Vec<String>,
    mut child_reader: os_pipe::PipeReader,
    mut child_writer: os_pipe::PipeWriter,
) -> i32 {
    match unsafe { libc::fork() as i32 } {
        0 => {
            //set up command to execute and initialize read buffer
            let mut buf = [0];
            let mut comm = Command::new(&command[0]);
            comm.args(&command[1..]);

            // Tell parent program child is set up to execute
            child_writer.write_all(&[1]).unwrap();
            drop(child_writer);

            //hear from parent that counters are set up
            let nread = child_reader.read(&mut buf).unwrap();
            assert_eq!(nread, 1);

            let e = comm.exec();
            panic!("child command failed: {}", e);
        }
        pid_child => pid_child,
    }
}

// Dummy function for parent test with subtests
fn dummy(_settings: &RunSettings) -> TestResult {
    TestResult::Passed
}

pub fn test_counts() -> Test {
    // This rets fail result, with verbose message if flag is on
    fn fail(x: String, settings: &RunSettings) -> TestResult {
        if settings.verbose {
            return TestResult::Failed(x);
        }
        TestResult::Failed("(1)".to_string())
    }
    fn event_counter(
        event_to_run: StatEvent,
        sane_number: isize,
        settings: &RunSettings,
    ) -> TestResult {
        let command_to_count = "cat".to_string();
        let command_args = "/dev/null".to_string();
        let (reader, mut writer) = pipe().unwrap();
        let (mut parent_reader, parent_writer) = pipe().unwrap();
        let child_reader = reader.try_clone().unwrap();
        let child_writer = parent_writer.try_clone().unwrap();
        let pid_child = launch_test_process(
            vec![String::from(&command_to_count), String::from(&command_args)],
            child_reader,
            child_writer,
        );
        let start: isize;
        let stop: isize;
        let event = Event::new(event_to_run, Some(pid_child));
        let mut buf = [0];
        let nread = parent_reader.read(&mut buf).unwrap();
        if nread != 1 {
            return fail(
                format!(
                    "\nINFO:\tresult from reader should be 1, instead got {}",
                    nread
                ),
                settings,
            );
        }
        start = event.start_counter().unwrap();
        writer.write_all(&[1]).unwrap();
        drop(writer);

        // wait for process to exit
        let mut status: libc::c_int = 0;
        let result = unsafe { libc::waitpid(pid_child, (&mut status) as *mut libc::c_int, 0) };
        if result != pid_child {
            return fail(
                format!(
                    "\nINFO:\tresult from call should be the same as pid child,
                     which was {}, instead got {}",
                    pid_child, result
                ),
                settings,
            );
        }
        stop = event.stop_counter().unwrap();
        let count = stop - start;
        if count < sane_number {
            return fail(
                format!(
                    "\nINFO:\tThe command {} had a count of {}, which was
                    less than the sane expected count of {}.",
                    command_to_count, count, sane_number
                ),
                settings,
            );
        }
        TestResult::Passed
    }

    fn test_cycles() -> Test {
        fn cycles(settings: &RunSettings) -> TestResult {
            event_counter(StatEvent::Cycles, 1000, settings)
        }
        Test {
            name: "cycles_test".to_string(),
            description: "Checks if cycles are over 1000".to_string(),
            call: cycles,
            subtests: Vec::new(),
            is_subtest: true,
        }
    }
    fn test_instructions() -> Test {
        fn instructions(settings: &RunSettings) -> TestResult {
            event_counter(StatEvent::Instructions, 1000, settings)
        }
        Test {
            name: "instructions_test".to_string(),
            description: "Checks if instructions are over 1000".to_string(),
            call: instructions,
            subtests: Vec::new(),
            is_subtest: true,
        }
    }
    fn test_context_switches() -> Test {
        fn context_switches(settings: &RunSettings) -> TestResult {
            event_counter(StatEvent::ContextSwitches, 0, settings)
        }
        Test {
            name: "context_switch_test".to_string(),
            description: "Checks if context switches are over 0".to_string(),
            call: context_switches,
            subtests: Vec::new(),
            is_subtest: true,
        }
    }
    fn test_l1d_cache_read() -> Test {
        fn l1d_cache_read(settings: &RunSettings) -> TestResult {
            event_counter(StatEvent::L1DCacheRead, 1000, settings)
        }
        Test {
            name: "L1D_cache_read_test".to_string(),
            description: "Checks if L1D cache read counts are over 1000".to_string(),
            call: l1d_cache_read,
            subtests: Vec::new(),
            is_subtest: true,
        }
    }
    fn test_l1d_cache_write() -> Test {
        fn l1d_cache_write(settings: &RunSettings) -> TestResult {
            event_counter(StatEvent::L1DCacheWrite, 0, settings)
        }
        Test {
            name: "L1D_cache_write_test".to_string(),
            description: "Checks if L1D cache write counts are over 0".to_string(),
            call: l1d_cache_write,
            subtests: Vec::new(),
            is_subtest: true,
        }
    }
    fn test_l1d_cache_read_misses() -> Test {
        fn l1d_cache_read_misses(settings: &RunSettings) -> TestResult {
            event_counter(StatEvent::L1DCacheReadMiss, 0, settings)
        }
        Test {
            name: "L1D_cache_read_miss_test".to_string(),
            description: "Checks if L1D cache read miss counts are over 0".to_string(),
            call: l1d_cache_read_misses,
            subtests: Vec::new(),
            is_subtest: true,
        }
    }
    fn test_l1i_cache_read_misses() -> Test {
        fn l1i_cache_read_misses(settings: &RunSettings) -> TestResult {
            event_counter(StatEvent::L1ICacheReadMiss, 0, settings)
        }
        Test {
            name: "L1I_cache_read_miss_test".to_string(),
            description: "Checks if L1I cache read miss counts are over 0".to_string(),
            call: l1i_cache_read_misses,
            subtests: Vec::new(),
            is_subtest: true,
        }
    }
    Test {
        name: "counts_test".to_string(),
        description: "Counts \"cat /dev/null\" for events sanity check".to_string(),
        call: dummy,
        subtests: vec![
            test_cycles(),
            test_instructions(),
            test_context_switches(),
            test_l1d_cache_read(),
            test_l1d_cache_write(),
            test_l1d_cache_read_misses(),
            test_l1i_cache_read_misses(),
        ],
        is_subtest: false,
    }
}
