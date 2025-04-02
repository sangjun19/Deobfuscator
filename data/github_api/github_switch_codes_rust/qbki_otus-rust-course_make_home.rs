// Repository: qbki/otus-rust-course
// File: smart-home-7/src/mocks/make_home.rs

use crate::common::{Device, SwitchStatusEnum::*};
use crate::smart_home::SmartHome;
use crate::smart_outlet::SmartOutlet;
use crate::smart_thermometer::SmartThermometer;

pub const KITCHEN: &str = "Kitchen";
pub const LIVING_ROOM: &str = "Living room";
pub const BASEMENT: &str = "Deep scary basement";

pub const UNKNOWN_OUTLET: &str = "Unknown outlet";

pub fn make_home() -> SmartHome {
    let fridge_outlet = SmartOutlet::new("Fridge", "127.0.0.1:20001");
    fridge_outlet.set_power(2000.0);
    fridge_outlet.set_switch(On);

    let unknown_outlet = SmartOutlet::new(UNKNOWN_OUTLET, "127.0.0.1:20002");
    unknown_outlet.set_power(1000.0);
    unknown_outlet.set_switch(Off);

    let outside_thermometer = SmartThermometer::new("Outside", "127.0.0.1:20101");
    outside_thermometer.set_temperature(30.0);

    let mut home = SmartHome::new("Home, sweet home");
    home.add_device(KITCHEN, Device::Outlet(fridge_outlet));
    home.add_device(LIVING_ROOM, Device::Thermometer(outside_thermometer));
    home.add_device(BASEMENT, Device::Outlet(unknown_outlet));

    home
}
