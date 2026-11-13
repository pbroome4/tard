//! Small helpers shared across modules.

use std::any::Any;

// normalizes probability vector to 1.0.
// Assumes sum is non-zero and that each element >=0
pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let sum:f32 = v.iter().sum();
    for e in v.iter_mut(){
        *e /= sum;
    }
    return v;
}

// Best-effort text of a panic payload (from JoinHandle::join or catch_unwind).
pub fn panic_message(payload: &Box<dyn Any + Send>) -> &str{
    if let Some(s) = payload.downcast_ref::<&str>(){
        return s;
    }
    if let Some(s) = payload.downcast_ref::<String>(){
        return s;
    }
    return "<non-string panic payload>";
}


#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn normalize_scales_to_sum_one(){
        assert_eq!(normalize(vec![1.0, 3.0]), vec![0.25, 0.75]);
    }

    #[test]
    fn panic_message_reads_str_and_string_payloads(){
        let p = std::panic::catch_unwind(|| panic!("static msg")).unwrap_err();
        assert_eq!(panic_message(&p), "static msg");
        let p = std::panic::catch_unwind(|| panic!("formatted {}", 7)).unwrap_err();
        assert_eq!(panic_message(&p), "formatted 7");
    }
}
