// Repository: mobley-trent/codewars-rust
// File: src/kata/spoonerize_me.rs

// 7 Kyu: Spoonerize Me
// https://www.codewars.com/kata/56b8903933dbe5831e000c76/train/rust

#[allow(dead_code)]
pub fn spoonerize(words: &str) -> String {
    let mut split_words = words.split_whitespace();
    let stream = split_words.clone();

    let a = split_words.next().unwrap();
    let b = split_words.last().unwrap();

    let (a_new, b_new) = switch_first_letters(a, b);
    let mut replaced: Vec<&str> = Vec::new();


    for word in stream.into_iter() {
        if *word == *a {
            replaced.push(&a_new)
        } else if *word == *b {
            replaced.push(&b_new)
        } else {
            replaced.push(word)
        }
    }

    replaced.join(" ")
}

fn switch_first_letters(str1: &str, str2: &str) -> (String, String) {
    let mut chars1 = str1.chars();
    let mut chars2 = str2.chars();

    let first_letter1 = chars1.next().unwrap();
    let first_letter2 = chars2.next().unwrap();

    let remaining1: String = chars1.collect();
    let remaining2: String = chars2.collect();

    let swapped1 = format!("{}{}", first_letter2, remaining1);
    let swapped2 = format!("{}{}", first_letter1, remaining2);

    (swapped1, swapped2)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tests() {
        assert_eq!(spoonerize("nit picking"), "pit nicking");
        assert_eq!(spoonerize("wedding bells"), "bedding wells");
        assert_eq!(spoonerize("jelly beans"), "belly jeans");
        assert_eq!(spoonerize("pack of lies"), "lack of pies");
    }
}