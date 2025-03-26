// Repository: Mario-Kart-Felix/Crispr-cas9-gene-editing-bot
// File: Rescripting.rs

let toBusy = v =>
  switch v {
  | Init => Loading              
  | Loading as a => a 
  | Loading rescript as a => a           
  | Reloading(_) as a => a
  | Complete(a) => Reloading(a)
}
