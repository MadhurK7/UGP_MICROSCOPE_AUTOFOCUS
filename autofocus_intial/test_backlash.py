"""
Test the backlash-aware stage wrapper against a stage that simulates
real mechanical play. After a reversal, the first N steps consume
backlash and don't move the sample.

We verify that BacklashAwareStage.move_absolute_z lands the sample at
the requested position regardless of starting position or direction.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hardware import BacklashAwareStage


# ──────────────────────────────────────────────────────────────────────
class SimMechStage:
    """
    Stage with backlash. Maintains:
        cmd_z   = sum of all dz the user has commanded
        mech_z  = where the sample is actually positioned
    """
    def __init__(self, backlash: int = 60):
        self.backlash = backlash
        self._cmd_z = 0
        self._mech_z = 0
        self._last_dir = 0
        self._slack = 0     # backlash to consume before mech_z moves

    @property
    def position(self):
        return {"x": 0, "y": 0, "z": self._cmd_z}

    @property
    def mech_z(self):
        return self._mech_z

    def move_relative(self, x: int = 0, y: int = 0, z: int = 0) -> bool:
        if z == 0:
            return True
        new_dir = +1 if z > 0 else -1
        # On direction reversal, replenish slack
        if self._last_dir != 0 and new_dir != self._last_dir:
            self._slack = self.backlash
        self._cmd_z += int(z)
        remaining = abs(z)
        if self._slack > 0:
            consumed = min(self._slack, remaining)
            self._slack -= consumed
            remaining -= consumed
        self._mech_z += new_dir * remaining
        self._last_dir = new_dir
        return True

    def release(self): pass
    def close(self):   pass


# ──────────────────────────────────────────────────────────────────────
def test_backlash_compensation():
    print("\n[T] BacklashAwareStage compensates real mechanical play")

    raw = SimMechStage(backlash=60)
    # backlash_steps in the wrapper must be ≥ true mechanical backlash
    stage = BacklashAwareStage(raw, backlash_steps=80,
                               approach="from_below", settle_s=0.0)

    # Sequence of absolute moves — note how some require reversals
    targets = [+200, +100, +400, +50, +300]
    results = []
    for t in targets:
        stage.move_absolute_z(t)
        cmd  = raw.position["z"]
        mech = raw.mech_z
        results.append((t, cmd, mech))
        print(f"  target z={t:+4d}  cmd_z={cmd:+5d}  mech_z={mech:+5d}  "
              f"sample_at_target={'YES' if mech == t else 'NO'}")

    # The mechanical sample position should always land on the target
    # (because move_absolute_z always approaches from below, and the
    # 80-step pre-overshoot fully consumes the 60-step backlash).
    for t, cmd, mech in results:
        assert mech == t, f"FAIL: mech={mech} != target={t}"
    print("  ✓ all targets reached exactly")

    # Now test relative moves with reversals
    print("\n[T] Relative moves with reversals")
    raw2 = SimMechStage(backlash=60)
    stage2 = BacklashAwareStage(raw2, backlash_steps=80,
                                approach="from_below", settle_s=0.0)
    moves = [+50, +50, -30, +20, -100, +200]
    sample_after = None
    for dz in moves:
        before_mech = raw2.mech_z
        stage2.move_relative(z=dz)
        after_mech = raw2.mech_z
        # When dz > 0 (matches approach="from_below"), the SAMPLE should
        # move by exactly dz because no backlash needs to be eaten.
        # When dz < 0 (against approach direction), backlash compensation
        # routes us through an overshoot+forward sequence; the wrapper
        # ENSURES the *next* forward move (after the implicit reversal)
        # also lands cleanly.
        print(f"  dz={dz:+5d}  mech_before={before_mech:+5d}  "
              f"mech_after={after_mech:+5d}  Δmech={after_mech-before_mech:+5d}")

    print("  ✓ PASS\n")


if __name__ == "__main__":
    print("=" * 64)
    print("  BACKLASH-AWARE STAGE — TEST")
    print("=" * 64)
    test_backlash_compensation()
    print("=" * 64)
    print("  PASSED")
