// Repository: gsmlg-dev/yew-duskmoon-ui
// File: packages/duskmoon-icons/src/mdi/icon_camera_switch.rs

#![allow(non_camel_case_types)]

use yew::prelude::*;
use super::props::IconProps;

#[function_component(MD_CameraSwitch)]
pub fn r#icon_camera_switch(props: &IconProps) -> Html {
  let owned_props = props.clone();

  html! {
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" id={owned_props.id} class={owned_props.class} width={owned_props.size} fill={owned_props.color} style={owned_props.style}>
      <path d="M15,15.5V13H9V15.5L5.5,12L9,8.5V11H15V8.5L18.5,12M20,4H16.83L15,2H9L7.17,4H4A2,2 0 0,0 2,6V18A2,2 0 0,0 4,20H20A2,2 0 0,0 22,18V6C22,4.89 21.1,4 20,4Z" />
    </svg>
  }
}
