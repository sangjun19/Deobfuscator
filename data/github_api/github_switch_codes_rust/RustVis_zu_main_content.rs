// Repository: RustVis/zu
// File: crates/zu-docs/src/components/main_content.rs

// Copyright (c) 2024 Xu Shaohua <shaohua@biofan.org>. All rights reserved.
// Use of this source is governed by Lesser General Public License
// that can be found in the LICENSE file.

use yew::prelude::*;
use yew_router::Switch;

use crate::components::footer::Footer;
use crate::components::header::Header;
use crate::components::left_panel::LeftPanel;
use crate::router::{switch_route, Route};

#[function_component(MainContent)]
pub fn main_content() -> Html {
    html! {
        <>
            <Header />
            <div class="container">
                <LeftPanel />
                <div class="col-sm-8 col-lg-10">
                    <Switch<Route> render={switch_route} />
                </div>
            </div>
            <Footer />
        </>
    }
}
