use std::collections::HashMap;
impl Solution {

    pub fn length_of_longest_substring(s: String) -> i32 {
    let mut char_indices = HashMap::new();
    let mut max_length = 0;
    let mut start = 0;

    for (i, c) in s.chars().enumerate() {
        if let Some(&prev_index) = char_indices.get(&c) {
            start = std::cmp::max(start, prev_index + 1);
        }
        max_length = std::cmp::max(max_length, i - start + 1);
        char_indices.insert(c, i);
    }

    max_length as i32
}
}