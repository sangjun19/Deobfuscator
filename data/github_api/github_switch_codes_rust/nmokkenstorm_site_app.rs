// Repository: nmokkenstorm/site
// File: src/app.rs

use yew::prelude::*;
use yew_router::prelude::*;

use crate::partials::{Footer, Header};
use crate::routes::{switch, Route};

#[function_component]
pub fn App() -> Html {
    html! {
      <BrowserRouter>
        <Header/>
        <main>
          <Switch<Route> render={switch} />
        </main>
        <Footer/>
      </BrowserRouter>
    }
}
