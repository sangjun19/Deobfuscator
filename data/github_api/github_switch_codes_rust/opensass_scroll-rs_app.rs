// Repository: opensass/scroll-rs
// File: examples/yew/src/app.rs

use yew::prelude::*;
use yew_router::prelude::*;

use crate::router::{switch, Route};

#[function_component(App)]
pub fn app() -> Html {
    html! {
      <BrowserRouter>
           <Switch<Route> render={switch} />
      </BrowserRouter>
    }
}
