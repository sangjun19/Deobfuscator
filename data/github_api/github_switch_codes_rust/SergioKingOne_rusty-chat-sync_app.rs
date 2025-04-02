// Repository: SergioKingOne/rusty-chat-sync
// File: src/components/app.rs

use crate::components::chat::Chat;
use crate::components::login::Login;
use crate::components::signup::SignUp;
use crate::services::auth::AuthService;
use crate::state::auth_state::{AuthAction, AuthState};
use yew::prelude::*;

#[function_component(App)]
pub fn app() -> Html {
    let auth_state = use_reducer(|| {
        if let Some(stored_auth) = AuthService::get_stored_auth() {
            AuthState {
                is_authenticated: true,
                token: Some(stored_auth.id_token),
                user_id: Some(stored_auth.username),
                error: None,
            }
        } else {
            AuthState {
                is_authenticated: false,
                token: None,
                user_id: None,
                error: None,
            }
        }
    });

    let show_signup = use_state(|| false);
    let selected_user = use_state(|| None::<String>);

    html! {
        if !auth_state.is_authenticated {
            if *show_signup {
                <SignUp
                    auth_state={auth_state.clone()}
                    on_switch_to_login={
                        let show_signup = show_signup.clone();
                        Callback::from(move |_| show_signup.set(false))
                    }
                />
            } else {
                <Login
                    auth_state={auth_state.clone()}
                    on_switch_to_signup={
                        let show_signup = show_signup.clone();
                        Callback::from(move |_| show_signup.set(true))
                    }
                />
            }
        } else {
            <Chat
                auth_state={auth_state.clone()}
                on_logout={
                    let auth_state = auth_state.clone();
                    Callback::from(move |_| auth_state.dispatch(AuthAction::Logout))
                }
                selected_user={(*selected_user).clone()}
                on_select_user={
                    let selected_user = selected_user.clone();
                    Callback::from(move |user| selected_user.set(user))
                }
            />
        }
    }
}
