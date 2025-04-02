// Repository: turboflakes/site
// File: app/src/router.rs

use crate::components::nav::Footer;
use yew::{function_component, html, Html};
use yew_router::{HashRouter, Routable, Switch};

use crate::pages::{
    about::AboutPage, home::HomePage, page_not_found::PageNotFound, privacy::PrivacyPage,
    terms::TermsPage,
};

#[function_component(Index)]
pub fn index() -> Html {
    html! {
        <HashRouter>
            <div class="mx-auto h-screen overflow-auto">
                <div class="bg-paper-texture">
                    <div class="grid grid-cols-8 gap-4">
                        <Switch<Routes> render={switch} />
                    </div>
                    <Footer />
                </div>
            </div>
        </HashRouter>
    }
}

#[derive(Routable, PartialEq, Eq, Clone, Debug)]
pub enum Routes {
    #[at("/")]
    Home,
    #[at("/about")]
    About,
    #[at("/terms")]
    Terms,
    #[at("/privacy")]
    Privacy,
    #[not_found]
    #[at("/404")]
    NotFound,
}

fn switch(routes: Routes) -> Html {
    match routes {
        Routes::Home => {
            html! { <HomePage /> }
        }
        Routes::About => {
            html! { <AboutPage /> }
        }
        Routes::Terms => {
            html! { <TermsPage /> }
        }
        Routes::Privacy => {
            html! { <PrivacyPage /> }
        }
        Routes::NotFound => {
            html! { <PageNotFound /> }
        }
    }
}
