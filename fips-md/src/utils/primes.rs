//! Prime number stuff

/// Prime factor decomposition
/// Panics on input 0, so watch out
pub(crate) fn prime_factors(n: usize) -> Vec<usize> {
    assert_ne!(n, 0);
    if n == 1 {
        return vec![];
    }
    let mut left = n;
    let mut factors = vec![];
    let mut i=2;
    while left >= i*i {
        while left % i == 0 {
            factors.push(i);
            left /= i;
        }
        i += 1;
    }
    if left > 1 {
        factors.push(left);
    }
    factors
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_prime_factors(){
        assert_eq!(prime_factors(1), vec![]);
        assert_eq!(prime_factors(2), vec![2]);
        assert_eq!(prime_factors(3), vec![3]);
        assert_eq!(prime_factors(4), vec![2,2]);
        assert_eq!(prime_factors(5), vec![5]);
        assert_eq!(prime_factors(6), vec![2,3]);
        assert_eq!(prime_factors(7), vec![7]);
        assert_eq!(prime_factors(8), vec![2,2,2]);
        assert_eq!(prime_factors(9), vec![3,3]);
        assert_eq!(prime_factors(10), vec![2,5]);
        assert_eq!(prime_factors(42), vec![2,3,7]);
    }
}