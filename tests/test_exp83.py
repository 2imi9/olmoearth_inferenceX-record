"""exp83's estimator and its population fact, checked by a second route (docs/plan/consensus_reliability.md).

An EM for Dawid-Skene written here, without the package, must agree with `reliability.dawid_skene` on a panel of
conditionally independent raters and both must recover the raters' true accuracies; the majority-shared error
share must equal a brute-force count."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))


def _ds_independent(votes, C, iters=50):
    """Dawid and Skene (1979) EM, written from the paper: E-step posteriors, M-step priors and confusions."""
    N, R = votes.shape
    T = np.zeros((N, C))
    for j in range(R):
        T[np.arange(N), votes[:, j]] += 1
    T /= T.sum(1, keepdims=True)
    for _ in range(iters):
        p = T.mean(0)
        pi = np.zeros((R, C, C)) + 0.01
        for j in range(R):
            for l in range(C):
                pi[j, :, l] += T[votes[:, j] == l].sum(0)
        pi /= pi.sum(2, keepdims=True)
        logT = np.log(p)[None, :].repeat(N, 0)
        for j in range(R):
            logT += np.log(pi[j][:, votes[:, j]]).T
        logT -= logT.max(1, keepdims=True)
        T = np.exp(logT)
        T /= T.sum(1, keepdims=True)
    p = T.mean(0)
    rel = np.array([(p * np.diag(pi[j])).sum() for j in range(R)])
    return T, pi, rel


def test_two_dawid_skene_implementations_agree_and_recover_independent_raters():
    from oe_inferencex.reliability import dawid_skene
    rng = np.random.default_rng(3)
    N, C, accs = 15000, 5, [0.92, 0.88, 0.8, 0.75, 0.6]
    y = rng.integers(0, C, N)
    votes = np.stack([np.where(rng.random(N) < a, y, (y + rng.integers(1, C, N)) % C) for a in accs], 1)
    _, _, rel_pkg = dawid_skene(votes, C)
    _, _, rel_ind = _ds_independent(votes, C)
    assert np.allclose(rel_pkg, rel_ind, atol=2e-3)
    assert np.abs(rel_pkg - np.array(accs)).max() < 0.02
    assert np.argsort(rel_pkg).tolist() == np.argsort(accs).tolist()


def test_majority_shared_share_equals_a_brute_force_count():
    import exp83_consensus as e83
    rng = np.random.default_rng(1)
    N, R, C = 3000, 5, 3
    y = rng.integers(0, C, N)
    votes = np.where(rng.random((N, R)) < 0.7, y[:, None], rng.integers(0, C, (N, R)))
    got = e83.majority_shared_share(votes, y)
    for j in range(R):
        n_err = n_same = 0
        for i in range(N):
            if votes[i, j] == y[i]:
                continue
            n_err += 1
            same = sum(1 for k in range(R) if k != j and votes[i, k] == votes[i, j])
            n_same += same > (R - 1) / 2
        assert abs(got[j] - n_same / n_err) < 1e-12


def test_correlated_raters_hide_their_shared_errors_from_dawid_skene():
    """Five raters that share a common error on a fixed fifth of the items: the hidden share tracks the
    majority-shared share, which is the mechanism P1 preregistered. The audit noted this case is unanimous on its
    shared errors, so it exercises only the credit (errors read as right); the next test adds the debit."""
    import exp83_consensus as e83
    from oe_inferencex.reliability import dawid_skene
    rng = np.random.default_rng(5)
    N, C = 20000, 4
    y = rng.integers(0, C, N)
    shared = rng.random(N) < 0.2
    wrong = (y + 1) % C
    votes = np.stack([np.where(shared, wrong, np.where(rng.random(N) < 0.9, y, (y + rng.integers(1, C, N)) % C)) for _ in range(5)], 1)
    acc = (votes == y[:, None]).mean(0)
    _, _, rel, info = dawid_skene(votes, C, return_info=True)
    assert info["converged"]
    hidden = (rel - acc) / (1 - acc)
    m = e83.majority_shared_share(votes, y)
    assert np.abs(hidden - m).max() < 0.10 and m.min() > 0.5


def test_a_rater_that_is_right_alone_is_debited_so_its_hidden_share_falls_below_the_shared_share():
    """exp83's finding: Dawid-Skene infers the plurality, so a strong rater that is right where a majority of the
    others agree on a wrong label is read as wrong there (a debit), and its hidden share sits below the share of
    its own errors the majority shares. Four weak raters share a wrong label on a fifth of the items; the fifth
    rater is right there and errs alone elsewhere."""
    import exp83_consensus as e83
    from oe_inferencex.reliability import dawid_skene
    rng = np.random.default_rng(6)
    N, C = 20000, 4
    y = rng.integers(0, C, N)
    shared = rng.random(N) < 0.2
    wrong = (y + 1) % C
    weak = [np.where(shared, wrong, np.where(rng.random(N) < 0.9, y, (y + rng.integers(1, C, N)) % C)) for _ in range(4)]
    strong = np.where(rng.random(N) < 0.92, y, (y + rng.integers(1, C, N)) % C)          # errs alone, never with the group
    votes = np.stack(weak + [strong], 1)
    acc = (votes == y[:, None]).mean(0)
    _, _, rel, info = dawid_skene(votes, C, return_info=True)
    assert info["converged"]
    hidden = (rel - acc) / (1 - acc)
    m = e83.majority_shared_share(votes, y)
    assert acc[4] == acc.max()                                   # the strong rater is the true best
    assert m[4] < 0.10 and hidden[4] < m[4] - 0.10               # little shared, yet debited: hidden well below shared
    assert rel[4] < acc[4]                                       # DS under-rates the best rater
    assert np.argmax(rel) != 4                                   # and does not pick it as best
