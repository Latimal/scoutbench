#!/usr/bin/env python3
"""Custom Expected Threat (xT) implementation for SPADL action data.

Reimplements the xT framework from scratch without socceraction's pandera
dependency (which is incompatible with NumPy 2.x). Uses value iteration on
a 16x12 grid over a 105x68m pitch.

The xT model values ball-moving actions (pass, dribble, cross) by computing
the difference in long-term scoring probability between start and end
locations. Higher xT = action moved the ball to a more dangerous position.

Reference: Karun Singh, "Introducing Expected Threat (xT)", 2019.
    https://karun.in/blog/expected-threat.html

Usage:
    from football_embed.data.xt_values import ExpectedThreat

    xt = ExpectedThreat()
    xt.fit(spadl_df)
    ratings = xt.rate(spadl_df)  # per-action xT delta (NaN for non-moves)
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

# SPADL standard pitch dimensions (meters)
FIELD_LENGTH: float = 105.0
FIELD_WIDTH: float = 68.0

# Grid resolution (columns x rows)
N: int = 16  # x-axis cells
M: int = 12  # y-axis cells

# SPADL action types that move the ball
_MOVE_TYPES = {"pass", "dribble", "cross"}

# SPADL shot types
_SHOT_TYPES = {"shot", "shot_freekick", "shot_penalty"}


def _cell_indexes(
    x: pd.Series, y: pd.Series, l: int = N, w: int = M
) -> tuple[pd.Series, pd.Series]:
    """Convert pitch coordinates to grid cell indices.

    Returns (xi, yj) where xi is the column index and yj is the row index.
    Origin is top-left of the grid (y=0 is top, y=w-1 is bottom).
    """
    xi = (x / FIELD_LENGTH * l).astype("int64").clip(0, l - 1)
    yj = (y / FIELD_WIDTH * w).astype("int64").clip(0, w - 1)
    return xi, yj


def _flat_indexes(
    x: pd.Series, y: pd.Series, l: int = N, w: int = M
) -> pd.Series:
    """Convert pitch coordinates to flat grid cell index.

    Grid layout: row 0 = top of pitch (high y), so we flip y.
    Flat index = (w - 1 - yj) * l + xi
    """
    xi, yj = _cell_indexes(x, y, l, w)
    return (w - 1 - yj) * l + xi


def _count_matrix(
    x: pd.Series, y: pd.Series, l: int = N, w: int = M
) -> npt.NDArray[np.int64]:
    """Count actions per grid cell, returns (w, l) matrix."""
    valid = ~(np.isnan(x) | np.isnan(y))
    flat = _flat_indexes(x[valid], y[valid], l, w)
    vc = flat.value_counts(sort=False)
    vec = np.zeros(w * l, dtype=np.int64)
    vec[vc.index] = vc.values
    return vec.reshape((w, l))


def _safe_divide(
    a: npt.ArrayLike, b: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Element-wise division, returning 0 where denominator is 0."""
    return np.divide(
        a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0, casting="unsafe"
    )


class ExpectedThreat:
    """Expected Threat (xT) model using value iteration on a grid.

    Parameters
    ----------
    l : int
        Grid cells along x-axis (pitch length). Default 16.
    w : int
        Grid cells along y-axis (pitch width). Default 12.
    eps : float
        Convergence threshold for value iteration. Default 1e-5.

    Attributes
    ----------
    xT : np.ndarray, shape (w, l)
        The converged xT surface. Higher values = more dangerous zones.
    """

    def __init__(self, l: int = N, w: int = M, eps: float = 1e-5) -> None:
        self.l = l
        self.w = w
        self.eps = eps
        self.xT: npt.NDArray[np.float64] = np.zeros((w, l))

    def fit(self, actions: pd.DataFrame) -> "ExpectedThreat":
        """Fit the xT model from SPADL actions.

        Expects columns: type_name, result_name, start_x, start_y, end_x, end_y.
        """
        l, w = self.l, self.w

        # 1. Scoring probability per cell
        shots = actions[actions["type_name"].isin(_SHOT_TYPES)]
        goals = shots[shots["result_name"] == "success"]
        shot_counts = _count_matrix(shots["start_x"], shots["start_y"], l, w)
        goal_counts = _count_matrix(goals["start_x"], goals["start_y"], l, w)
        scoring_prob = _safe_divide(goal_counts, shot_counts)

        # 2. Action choice probabilities (shoot vs move)
        moves = actions[actions["type_name"].isin(_MOVE_TYPES)]
        move_counts = _count_matrix(moves["start_x"], moves["start_y"], l, w)
        total_counts = move_counts + shot_counts
        p_shot = _safe_divide(shot_counts, total_counts)
        p_move = _safe_divide(move_counts, total_counts)

        # 3. Transition matrix for moves (flat_from -> flat_to)
        successful_moves = moves[moves["result_name"] == "success"]
        start_flat = _flat_indexes(moves["start_x"], moves["start_y"], l, w)
        start_vc = start_flat.value_counts(sort=False)
        start_counts_flat = np.zeros(w * l)
        start_counts_flat[start_vc.index] = start_vc.values

        # Build transition matrix
        sm_start = _flat_indexes(
            successful_moves["start_x"], successful_moves["start_y"], l, w
        )
        sm_end = _flat_indexes(
            successful_moves["end_x"], successful_moves["end_y"], l, w
        )

        transition = np.zeros((w * l, w * l), dtype=np.float64)
        for i in range(w * l):
            mask = sm_start == i
            if mask.sum() == 0:
                continue
            end_vc = sm_end[mask].value_counts(sort=False)
            transition[i, end_vc.index] = end_vc.values / start_counts_flat[i]

        # 4. Value iteration
        gs = scoring_prob * p_shot
        self.xT = np.zeros((w, l), dtype=np.float64)

        for _ in range(100):  # max iterations safety
            # Compute expected payoff from moving: for each cell,
            # sum over all destination cells of T(i,j) * xT(j)
            xT_flat = self.xT.flatten()
            payoff_flat = transition @ xT_flat  # (w*l,)
            payoff = payoff_flat.reshape((w, l))

            new_xT = gs + p_move * payoff
            diff = np.abs(new_xT - self.xT).max()
            self.xT = new_xT

            if diff < self.eps:
                break

        return self

    def rate(self, actions: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Compute per-action xT delta.

        Only successful move actions (pass, dribble, cross) get a value.
        All other actions get NaN.

        Returns array of length len(actions).
        """
        l, w = self.l, self.w
        grid = self.xT

        ratings = np.full(len(actions), np.nan)

        # Successful moves only
        is_move = actions["type_name"].isin(_MOVE_TYPES)
        is_success = actions["result_name"] == "success"
        mask = is_move & is_success
        move_idx = actions.index[mask]

        if len(move_idx) == 0:
            return ratings

        start_xi, start_yj = _cell_indexes(
            actions.loc[move_idx, "start_x"],
            actions.loc[move_idx, "start_y"],
            l, w,
        )
        end_xi, end_yj = _cell_indexes(
            actions.loc[move_idx, "end_x"],
            actions.loc[move_idx, "end_y"],
            l, w,
        )

        # Grid is (w, l) with row 0 = top of pitch (high y)
        start_row = w - 1 - start_yj
        end_row = w - 1 - end_yj

        xT_start = grid[start_row.values, start_xi.values]
        xT_end = grid[end_row.values, end_xi.values]

        # Use positional indexing to write back
        pos_idx = np.where(mask)[0]
        ratings[pos_idx] = xT_end - xT_start

        return ratings
