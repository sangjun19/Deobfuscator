// Repository: jam1garner/graph_percent
// File: src/main.rs

use smush_discord_shared::Info;
use std::net::{TcpStream, IpAddr};
use std::io::{BufRead, BufReader};
use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::prelude::*;

const IP_ADDR_FILE: &str = "ip_addr.txt";

fn get_home_ip_str() -> Option<String> {
    let switch_home_dir = dirs::home_dir()?.join(".switch");
    if switch_home_dir.exists() {
        let ip_addr_file = switch_home_dir.join(IP_ADDR_FILE);
        if ip_addr_file.exists() {
            std::fs::read_to_string(ip_addr_file).ok()
        } else {
            None
        }
    } else {
        None
    }
}

fn get_home_ip() -> IpAddr {
    let ip = get_home_ip_str().unwrap();
    dbg!(ip).trim().parse().unwrap()
}

fn get_info(bytes: &[u8]) -> Info {
    serde_json::from_slice(bytes).unwrap()
}

fn main() {
    let mut packets = BufReader::new(TcpStream::connect((get_home_ip(), 4242u16)).unwrap()).split(b'\n');
    let mut window: PistonWindow = WindowSettings::new("Real Time Percent Graph", [450, 300])
        .samples(4)
        .build()
        .unwrap();
    window.set_max_fps(10);
    let mut p1_percent = vec![];
    let mut p2_percent = vec![];

    loop {
        if let Some(packet) = packets.next() {
            let info = get_info(&packet.unwrap());
            if info.is_match() {
                p1_percent.push(info.players[0].damage());
                p2_percent.push(info.players[1].damage());
                break;
            }
        } else {
            println!("Connection ended before match started");
            return;
        }
    }

    while let Some(_) = draw_piston_window(&mut window, |b| {
        let info = get_info(&packets.next().ok_or("")??);
        p1_percent.push(info.players[0].damage());
        p2_percent.push(info.players[1].damage());
        let root = b.into_drawing_area();
        root.fill(&WHITE)?;
        let mut cc = ChartBuilder::on(&root)
            .margin(10)
            .caption("Real Time Percent Graph", ("sans-serif", 30).into_font())
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_ranged(0.0..p1_percent.len() as f32 * 2., 0f32..300.0)?;

        for (idx, data) in [&p1_percent, &p2_percent].iter().enumerate() {
            cc.draw_series(LineSeries::new(
                data.iter().enumerate().map(|(a, b)| (a as f32 * 0.5, *b)),
                &Palette99::pick(idx),
            ))?
            .label(format!("Player {}", idx))
            .legend(move |(x, y)| {
                Rectangle::new([(x - 5, y - 5), (x + 5, y + 5)], &Palette99::pick(idx))
            });
        }

        cc.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }) {}
}
