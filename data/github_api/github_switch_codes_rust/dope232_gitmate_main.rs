// Repository: dope232/gitmate
// File: src/main.rs

use std::process::{Command, exit};

fn commit_update_push(commit_message: &str) {
    let add_command = Command::new("git")
        .arg("add")
        .arg("-A")
        .output()
        .expect("Failed to execute git add command");

    if !add_command.status.success() {
        eprintln!("Failed to add files to the git repo");
        exit(1);
    }

    let commit_command = Command::new("git")
        .arg("commit")
        .arg("-m")
        .arg(commit_message)
        .output()
        .expect("Failed to execute git commit command");

    if !commit_command.status.success() {
        eprintln!("Error: Failed to commit changes.");
        exit(1);
    }

    let push_command = Command::new("git")
        .arg("push")
        .arg("-u")
        .arg("origin")
        .arg("master")
        .output()
        .expect("Failed to execute git push command");

    if !push_command.status.success() {
        eprintln!("Failed to push files to the git repo");
        exit(1);
    }

    println!("Successfully committed and pushed to the repo");
}

fn pull_origin_master() {
    let pull_command = Command::new("git")
        .arg("pull")
        .arg("origin")
        .arg("master")
        .output()
        .expect("Failed to execute git pull command");

    if !pull_command.status.success() {
        eprintln!("Failed to pull changes from origin/master");
        exit(1);
    }

    println!("Successfully pulled changes from origin/master");
}

fn branch_and_switch(branch_name: &str) {
    let branch_command = Command::new("git")
        .arg("branch")
        .arg(branch_name)
        .output()
        .expect("Failed to execute git branch command");

    if !branch_command.status.success() {
        eprintln!("Failed to create branch {}", branch_name);
        exit(1);
    }

    let checkout_command = Command::new("git")
        .arg("checkout")
        .arg(branch_name)
        .output()
        .expect("Failed to execute git checkout command");

    if !checkout_command.status.success() {
        eprintln!("Failed to switch to branch {}", branch_name);
        exit(1);
    }

    println!("Switched to branch {}", branch_name);

    // Push changes to the newly created branch
    commit_update_push_branch("Initial commit", branch_name);
}

fn commit_update_push_branch(commit_message: &str, branch_name: &str) {
    let add_command = Command::new("git")
        .arg("add")
        .arg("-A")
        .output()
        .expect("Failed to execute git add command");

    if !add_command.status.success() {
        eprintln!("Failed to add files to the git repo");
        exit(1);
    }

    let commit_command = Command::new("git")
        .arg("commit")
        .arg("-m")
        .arg(commit_message)
        .output()
        .expect("Failed to execute git commit command");

    if !commit_command.status.success() {
        eprintln!("Error: Failed to commit changes.");
        exit(1);
    }

    let push_command = Command::new("git")
        .arg("push")
        .arg("-u")
        .arg("origin")
        .arg(branch_name)
        .output()
        .expect("Failed to execute git push command");

    if !push_command.status.success() {
        eprintln!("Failed to push files to the git repo");
        exit(1);
    }

    println!("Successfully committed and pushed to the branch");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: gitmate <commit_message> or gitmate pull or gitmate branch <branch_name>");
        exit(1);
    }

    if args[1] == "pull" {
        pull_origin_master();
    } else if args[1] == "branch" { 
        if args.len() < 3 {
            eprintln!("Usage: gitmate branch <branch_name>");
            exit(1);
        }
        let branch_name = &args[2];
        branch_and_switch(branch_name);
    } else {
        let commit_message = &args[1];
        commit_update_push(commit_message);
    }
}
