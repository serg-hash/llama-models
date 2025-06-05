import mpmath as mp

mp.dps = 50  # high precision


def compute_zero(index):
    """Compute the nth nontrivial zero of the Riemann zeta function"""
    # get approximate zero on the critical line
    guess = mp.zetazero(index)
    # refine using complex root finder
    return mp.findroot(mp.zeta, guess)


def find_zeros(n=10):
    """Return a list with the first n zeros."""
    zeros = []
    for i in range(1, n + 1):
        zero = compute_zero(i)
        zeros.append(zero)
    return zeros


def check_critical_line(zero, tol=1e-10):
    """Check if zero lies on the critical line Re(s) = 1/2."""
    return abs(mp.re(zero) - mp.mpf('0.5')) < tol


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore zeros of the Riemann zeta function")
    parser.add_argument("-n", type=int, default=10, help="Number of zeros to compute")
    parser.add_argument("--tol", type=float, default=1e-10, help="Tolerance for critical line check")
    args = parser.parse_args()

    zeros = find_zeros(args.n)
    for idx, z in enumerate(zeros, start=1):
        print(f"Zero {idx}: {z}")
        print(f"On critical line: {check_critical_line(z, args.tol)}\n")

    all_on_line = all(check_critical_line(z, args.tol) for z in zeros)
    print(f"All computed zeros on critical line: {all_on_line}")
