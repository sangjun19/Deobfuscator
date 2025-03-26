// Repository: Mario-Kart-Felix/web4
// File: Rescript.rs

let toBusy = v =>
  switch v {
  | Init => Loading              
  | Loading as a => a 
  | Loading rescript as a => a           
  | Reloading(_) as a => a
  | Complete(a) => Reloading(a)
}
