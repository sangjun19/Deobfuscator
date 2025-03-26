// Repository: HesterG/pika-chat-project
// File: pika-chat-frontend/src/routes.rs

use yew::prelude::*;
use yew_router::prelude::*;
use crate::pages::login::Login;
use crate::pages::register::Register;
use crate::pages::dashboard::Dashboard;
use crate::pages::home::Home;
use crate::pages::chatroom::ChatRoom;

// Define your app's routes
#[derive(Clone, Routable, PartialEq)]
pub enum Route {
    #[at("/dashboard")]
    Dashboard,
    #[at("/login")]
    Login,
    #[at("/register")]
    Register,
    #[at("/chatroom/:room_id")]
    ChatRoom { room_id: i64 },
    #[at("/")]
    Home,
}

// Define the route switch function
pub fn switch(route: &Route) -> Html {
    match route {
        Route::Login => html! { <Login /> },
        Route::Register => html! { <Register /> },
        Route::Home => html! { <Home /> },
        Route::Dashboard => html! { <Dashboard /> },
        Route::ChatRoom { room_id } => html! { <ChatRoom room_id={*room_id} /> },
    }
}
