"""Tests for the maze bank (roadmap task t6).

Run from this directory:
    python -m pytest test_maze_bank.py -q
"""

import math

import numpy as np
import pytest
import yaml

import maze_bank as mb
from config import WALL_DOWN, WALL_LEFT, WALL_RIGHT, WALL_UP
from maze_bank import NONE

SIZE_SHAPES = mb.SIZE_SHAPES
PREDECESSOR_DIR = mb.MAZE_DIR / "predecessor"


def _index():
    return yaml.safe_load(mb.INDEX_PATH.read_text(encoding="utf-8"))["mazes"]


# --------------------------------------------------------------------------
# Stored bank
# --------------------------------------------------------------------------

def test_index_lists_exactly_the_stored_mazes():
    assert sorted(_index()) == mb.list_mazes()


@pytest.mark.parametrize("name", mb.list_mazes())
def test_every_maze_loads_and_validates(name):
    spec = mb.load_maze(name)
    assert spec.name == name
    assert spec.wall_map.dtype == np.uint8
    assert spec.wall_map.shape == (spec.rows, spec.cols, 4)
    assert mb.wall_map_problems(spec.wall_map) == []
    size = name.split("/")[0]
    assert (spec.rows, spec.cols) == SIZE_SHAPES[size]
    assert spec.goal_cells == mb.center_goal(spec.rows, spec.cols)
    for r, c in spec.goal_cells:
        assert 0 <= r < spec.rows and 0 <= c < spec.cols
    # The .txt drawing parses back to the same wall_map.
    txt = mb.maze_path(name).with_suffix(".txt")
    assert np.array_equal(mb.maze_to_wall_map(txt), spec.wall_map)


@pytest.mark.parametrize("size", mb.SIZES)
def test_at_least_three_mazes_per_size(size):
    names = mb.list_mazes(size)
    assert len(names) >= 3
    for name in names:
        assert mb.load_maze(name).wall_map.shape == SIZE_SHAPES[size] + (4,)


def test_runner_default_maze_exists():
    spec = mb.load_maze("medium/m10_a")
    assert (spec.rows, spec.cols) == (10, 10)
    assert spec.goal_cells == ((4, 4), (4, 5), (5, 4), (5, 5))


def test_bare_name_lookup_and_missing_name():
    assert mb.load_maze("m10_a").name == "medium/m10_a"
    with pytest.raises(KeyError, match="not found"):
        mb.load_maze("medium/does_not_exist")
    with pytest.raises(KeyError, match="not found"):
        mb.load_maze("does_not_exist")


def test_load_is_case_sensitive_on_every_os():
    # On Windows the file system would happily open MEDIUM/M10_A.npz; the name check must not.
    for bad in ("MEDIUM/m10_a", "medium/M10_A", "M10_A"):
        with pytest.raises(KeyError, match="not found"):
            mb.load_maze(bad)


def test_ambiguous_bare_name(tmp_path):
    for size in ("small", "custom"):
        rows, cols = (5, 5) if size == "small" else (3, 4)
        spec = mb.MazeSpec(f"{size}/dup", mb.generate_maze(rows, cols, 1), ((0, 0),), rows, cols, {})
        mb.save_maze(spec, root=tmp_path)
    with pytest.raises(KeyError, match="ambiguous"):
        mb.load_maze("dup", root=tmp_path)
    assert mb.load_maze("custom/dup", root=tmp_path).rows == 3


@pytest.mark.parametrize("bad", ["small/sub/y", "small", "/x", "small/", "small/../x", "small/a b", r"a\b/c"])
def test_bad_names_rejected(bad, tmp_path):
    spec = mb.MazeSpec(bad, mb.generate_maze(5, 5, 1), ((2, 2),), 5, 5, {})
    with pytest.raises(ValueError, match="maze name"):
        mb.save_maze(spec, root=tmp_path)
    with pytest.raises(KeyError):
        mb.load_maze(bad, root=tmp_path)


def test_load_returns_independent_copies():
    a = mb.load_maze("small/m5_a")
    a.wall_map[:] = 0
    assert mb.validate_wall_map(mb.load_maze("small/m5_a").wall_map)


@pytest.mark.parametrize("name", mb.list_mazes())
def test_index_matches_recomputed_ambiguity(name):
    entry = _index()[name]
    spec = mb.load_maze(name)
    assert entry["size"] == name.split("/")[0]
    assert [tuple(g) for g in entry["goal_cells"]] == list(spec.goal_cells)
    amb = entry["ambiguity"]
    assert {"unbounded", "R2"} <= set(amb)
    for key, value in amb["unbounded"].items():
        assert mb.ambiguity(spec.wall_map)["unbounded"][key] == pytest.approx(value, abs=1e-6), (name, key)
    for R_key, scores in amb.items():
        if R_key == "unbounded":
            continue
        got = mb.ambiguity(spec.wall_map, int(R_key.removeprefix("R")))["limited"]
        assert set(got) == set(scores)
        for key, value in scores.items():
            assert got[key] == pytest.approx(value, abs=1e-6), (name, R_key, key)


@pytest.mark.parametrize(
    "name", [n for n, e in _index().items() if e["source"]["kind"] == "generator"]
)
def test_generator_entries_regenerate_exactly(name):
    """Guards against numpy RNG stream changes: index seed + loop_fraction -> same wall_map."""
    src = _index()[name]["source"]
    spec = mb.load_maze(name)
    regenerated = mb.generate_maze(
        spec.rows, spec.cols, src["seed"], src["loop_fraction"], start=tuple(src["dfs_start"])
    )
    assert np.array_equal(regenerated, spec.wall_map), name


def test_difficulty_ties_share_a_label():
    index = _index()
    for size in mb.SIZES:
        by_h1 = {}
        for n in mb.list_mazes(size):
            by_h1.setdefault(round(index[n]["ambiguity"]["R2"]["H1"], 6), set()).add(index[n]["difficulty"])
        assert all(len(labels) == 1 for labels in by_h1.values()), (size, by_h1)
        # Labels are monotone in H1.
        order = {"easy": 0, "medium": 1, "hard": 2}
        ranked = [order[min(by_h1[h])] for h in sorted(by_h1)]
        assert ranked == sorted(ranked), size


def test_bank_spans_easy_to_hard():
    """Within each size the mazes are not all equally ambiguous."""
    index = _index()
    for size in mb.SIZES:
        h1 = [index[n]["ambiguity"]["R2"]["H1"] for n in mb.list_mazes(size)]
        assert max(h1) - min(h1) > 0.2, size
        labels = {index[n]["difficulty"] for n in mb.list_mazes(size)}
        assert {"easy", "hard"} <= labels, size


# --------------------------------------------------------------------------
# Generator and validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rows, cols", [(5, 5), (10, 10), (16, 16), (3, 7)])
@pytest.mark.parametrize("loop_fraction", [0.0, 0.2])
def test_generator_is_valid_and_deterministic(rows, cols, loop_fraction):
    a = mb.generate_maze(rows, cols, seed=123, loop_fraction=loop_fraction)
    b = mb.generate_maze(rows, cols, seed=123, loop_fraction=loop_fraction)
    assert a.shape == (rows, cols, 4)
    assert np.array_equal(a, b)
    assert mb.validate_wall_map(a)
    assert not np.array_equal(a, mb.generate_maze(rows, cols, seed=124, loop_fraction=loop_fraction))


def _n_open_interior(wall_map):
    rows, cols = wall_map.shape[:2]
    return sum(int(wall_map[r, c, w]) for r, c, w in mb.interior_edges(rows, cols))


def test_generator_loop_injection_counts():
    rows, cols = 10, 10
    n_interior = len(mb.interior_edges(rows, cols))
    perfect = mb.generate_maze(rows, cols, seed=5, loop_fraction=0.0)
    assert _n_open_interior(perfect) == rows * cols - 1  # spanning tree
    closed_after_dfs = n_interior - (rows * cols - 1)
    loopy = mb.generate_maze(rows, cols, seed=5, loop_fraction=0.25)
    assert _n_open_interior(loopy) == rows * cols - 1 + round(0.25 * closed_after_dfs)
    assert np.all(loopy >= perfect)  # loops only open walls
    assert np.array_equal(mb.generate_maze(rows, cols, seed=5, loop_fraction=1.0), mb.open_wall_map(rows, cols))


def test_generator_matches_project_a_dfs_on_15x20():
    map_generator = pytest.importorskip("map_generator")
    expected, _start, _goal = map_generator._dense_photo_maze(601, (7, 12), (6, 1))
    assert np.array_equal(mb.generate_maze(15, 20, 601, 0.0, start=(7, 12)), expected)


def test_validate_rejects_broken_maps():
    good = mb.open_wall_map(4, 4)
    assert mb.validate_wall_map(good)

    border = good.copy()
    border[0, 2, WALL_UP] = 1
    assert not mb.validate_wall_map(border)

    asym = good.copy()
    asym[1, 1, WALL_RIGHT] = 0  # the (1, 2) LEFT side stays open
    assert not mb.validate_wall_map(asym)

    split = good.copy()
    for r in range(4):
        mb.set_edge(split, r, 1, WALL_RIGHT, 0)
    assert not mb.validate_wall_map(split)

    assert not mb.validate_wall_map(np.ones((4, 4, 3), dtype=np.uint8))
    assert not mb.validate_wall_map([[1]])


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    wall_map = mb.generate_maze(6, 4, seed=9, loop_fraction=0.1)
    spec = mb.MazeSpec(
        name="custom/round_trip",
        wall_map=wall_map,
        goal_cells=((2, 1), (3, 2)),
        rows=6,
        cols=4,
        meta={"source": {"kind": "generator", "seed": 9}},
    )
    path = mb.save_maze(spec, root=tmp_path)
    assert path.exists() and path.with_suffix(".txt").exists()
    assert mb.list_mazes(root=tmp_path) == ["custom/round_trip"]
    back = mb.load_maze("custom/round_trip", root=tmp_path)
    assert np.array_equal(back.wall_map, wall_map)
    assert back.wall_map.dtype == np.uint8
    assert back.goal_cells == spec.goal_cells
    assert (back.rows, back.cols) == (6, 4)
    assert back.meta["source"] == {"kind": "generator", "seed": 9}


@pytest.mark.parametrize(
    "name, rows, cols, map_shape, goals, match",
    [
        ("custom/x", 4, 4, (4, 5), ((0, 0),), "does not match"),   # wall_map vs rows/cols
        ("small/x", 4, 4, (4, 4), ((0, 0),), "small mazes must be"),  # size class
        ("medium/x", 5, 5, (5, 5), ((2, 2),), "medium mazes must be"),
        ("custom/x", 4, 4, (4, 4), (), "goal_cells is empty"),
        ("custom/x", 4, 4, (4, 4), ((4, 0),), "outside"),
        ("custom/x", 4, 4, (4, 4), ((0, -1),), "outside"),
    ],
)
def test_save_rejects_inconsistent_specs(tmp_path, name, rows, cols, map_shape, goals, match):
    wall_map = mb.generate_maze(map_shape[0], map_shape[1], seed=2)
    spec = mb.MazeSpec(name, wall_map, goals, rows, cols, {})
    with pytest.raises(ValueError, match=match):
        mb.save_maze(spec, root=tmp_path, check=False)  # enforced even without check
    assert mb.list_mazes(root=tmp_path) == []


def test_save_refuses_invalid_maze(tmp_path):
    bad = mb.closed_wall_map(3, 3)
    spec = mb.MazeSpec("custom/bad", bad, ((1, 1),), 3, 3, {})
    with pytest.raises(ValueError):
        mb.save_maze(spec, root=tmp_path)


# --------------------------------------------------------------------------
# Ambiguity on hand-built mazes
# --------------------------------------------------------------------------

def test_ray_distances_open_3x3_box():
    d = mb.ray_distances(mb.open_wall_map(3, 3))
    # Channel order UP, DOWN, LEFT, RIGHT.
    assert tuple(d[0, 0]) == (0, 2, 0, 2)
    assert tuple(d[1, 1]) == (1, 1, 1, 1)
    assert tuple(d[2, 1]) == (2, 0, 1, 1)
    s1 = mb.sensor_signature(mb.open_wall_map(3, 3), R=1)
    assert tuple(s1[0, 0]) == (0, NONE, 0, NONE)
    assert tuple(s1[1, 1]) == (1, 1, 1, 1)


def test_ambiguity_open_3x3_box_all_unique():
    # In an open box every cell's (up, down, left, right) distances are (r, 2-r, c, 2-c).
    a = mb.ambiguity(mb.open_wall_map(3, 3), R=2)
    for kind in ("unbounded", "limited"):
        s = a[kind]
        assert s == {"n_cells": 9, "n_shared": 0, "n_distinct": 9, "amb": 0.0, "largest_group": 1, "H1": 0.0}
    # With R=0 a ray only tells "wall on my edge" or not: 9 cells, 3 row-classes x 3 col-classes...
    # rows: top (0,NONE), middle (NONE,NONE), bottom (NONE,0); same for cols -> 9 distinct.
    assert mb.ambiguity(mb.open_wall_map(3, 3), R=0)["limited"]["n_distinct"] == 9
    # ...but a 4x4 box at R=0 merges the 2x2 middle block (all NONE) and the edge pairs.
    s = mb.ambiguity(mb.open_wall_map(4, 4), R=0)["limited"]
    assert s["n_distinct"] == 9
    assert s["largest_group"] == 4
    assert s["n_shared"] == 4 + 4 * 2  # center block + four edge pairs


def test_ambiguity_1xN_corridor():
    n = 7
    corridor = mb.open_wall_map(1, n)
    a = mb.ambiguity(corridor, R=2)
    # Unbounded: (0, 0, c, n-1-c) is unique for every c.
    assert a["unbounded"]["n_shared"] == 0
    assert a["unbounded"]["n_distinct"] == n
    # R=2: left reading c if c<=2 else NONE, right likewise. c=0,1,2 and 4,5,6 are unique,
    # c=3 is (NONE, NONE) alone -> all 7 distinct.
    assert a["limited"]["n_distinct"] == 7
    # R=1: c=0,1 and 5,6 unique; c=2,3,4 all read (NONE, NONE).
    s = mb.ambiguity(corridor, R=1)["limited"]
    assert s["n_distinct"] == 5
    assert s["n_shared"] == 3
    assert s["largest_group"] == 3
    assert s["amb"] == pytest.approx(1 - 5 / 7)
    assert s["H1"] == pytest.approx((3 / 7) * math.log(3))


def test_ambiguity_comb_of_identical_dead_ends():
    # 2x4 "comb": a corridor along row 0 and four 1-cell teeth below it, separated by
    # vertical walls in row 1. Every tooth reads (up 1, down 0, left 0, right 0) at any
    # range, so the four teeth are indistinguishable by a single look.
    w = mb.open_wall_map(2, 4)
    for c in range(3):
        mb.set_edge(w, 1, c, WALL_RIGHT, 0)
    assert mb.validate_wall_map(w)

    # R=0 (own-edge walls only): teeth (4 cells) and the two middle corridor cells
    # (0,1), (0,2) (wall up only) are groups; the corridor ends (0,0), (0,3) are unique.
    sig0 = mb.sensor_signature(w, R=0)
    assert tuple(sig0[1, 0]) == tuple(sig0[1, 3]) == (NONE, 0, 0, 0)
    assert tuple(sig0[0, 1]) == tuple(sig0[0, 2]) == (0, NONE, NONE, NONE)
    s = mb.ambiguity(w, R=0)["limited"]
    assert s == {"n_cells": 8, "n_shared": 6, "n_distinct": 4, "amb": pytest.approx(0.5),
                 "largest_group": 4, "H1": pytest.approx((4 / 8) * math.log(4) + (2 / 8) * math.log(2))}

    # R=1 sees one cell further: (0,1) reads left 1 / right NONE, (0,2) the mirror -> unique.
    s = mb.ambiguity(w, R=1)["limited"]
    assert (s["n_shared"], s["n_distinct"], s["largest_group"]) == (4, 5, 4)
    assert tuple(mb.sensor_signature(w, R=1)[1, 2]) == (1, 0, 0, 0)
    # Unbounded (Project A) signature: the teeth stay shared too.
    assert mb.ambiguity(w)["unbounded"]["n_shared"] == 4


def test_ray_distances_match_project_a_helper():
    map_generator = pytest.importorskip("map_generator")
    w = mb.generate_maze(6, 7, seed=11, loop_fraction=0.3)
    d = mb.ray_distances(w)
    for r in range(6):
        for c in range(7):
            expected = tuple(
                map_generator._wall_distance(w, r, c, k) for k in (WALL_UP, WALL_DOWN, WALL_LEFT, WALL_RIGHT)
            )
            assert tuple(d[r, c]) == expected


def test_unbounded_matches_project_a_score_map_on_project_a_maps():
    map_generator = pytest.importorskip("map_generator")
    for m in map_generator.get_maps():
        score = map_generator.score_map(m["wall_map"], m["start"], m["goal"])
        assert mb.ambiguity(m["wall_map"])["unbounded"]["n_shared"] == score["ambiguity_score"], m["name"]


def test_group_stats_formulas():
    s = mb.group_stats([1, 2, 3])
    assert s["n_cells"] == 6
    assert s["n_shared"] == 5
    assert s["n_distinct"] == 3
    assert s["amb"] == pytest.approx(0.5)
    assert s["largest_group"] == 3
    assert s["H1"] == pytest.approx((2 / 6) * math.log(2) + (3 / 6) * math.log(3))
    # One group of everything: observation tells nothing, H1 = ln N.
    assert mb.group_stats([10])["H1"] == pytest.approx(math.log(10))


# --------------------------------------------------------------------------
# Predecessor .maze conversion
# --------------------------------------------------------------------------

def test_parse_simple_maze_hand_checked_cells():
    w = mb.maze_to_wall_map(PREDECESSOR_DIR / "simple.maze")
    assert w.shape == (5, 5, 4)
    assert mb.validate_wall_map(w)
    # Top line "|   |               |": (0,0) walls N, W, E; S open.
    assert tuple(w[0, 0]) == (0, 1, 0, 0)  # UP, DOWN, LEFT, RIGHT
    # (0,1): N wall, W wall (shared with (0,0)), E open, S open ("+   +   +---+").
    assert tuple(w[0, 1]) == (0, 1, 0, 1)
    # (0,2): N wall, S wall ("---" under col 2), E open, W open.
    assert tuple(w[0, 2]) == (0, 0, 1, 1)
    # (2,3): row "|               |   |": E wall; above "+   +---+---+   +   +": N open;
    # below "+   +---+   +   +   +": S open.
    assert tuple(w[2, 3]) == (1, 1, 1, 0)


def test_parse_inline_text_and_bits():
    text = "+---+---+\n|       |\n+   +---+\n|   |   |\n+---+---+\n"
    # 2x2: (1,1) is sealed off, so the maze is invalid, but the parse is still exact.
    bits = mb.parse_maze_text(text)
    assert bits.tolist() == [
        [mb.BIT_N | mb.BIT_W, mb.BIT_N | mb.BIT_E | mb.BIT_S],
        [mb.BIT_W | mb.BIT_E | mb.BIT_S, mb.BIT_N | mb.BIT_E | mb.BIT_S | mb.BIT_W],
    ]
    w = mb.maze_to_wall_map(text)
    assert tuple(w[0, 0]) == (0, 1, 0, 1)
    assert not mb.validate_wall_map(w)


def test_parse_japan_2017_hand_checked_cells():
    w = mb.maze_to_wall_map(PREDECESSOR_DIR / "japan_2017.maze")
    assert w.shape == (16, 16, 4)
    assert mb.validate_wall_map(w)
    # First rows of the drawing:
    # +---+---+ ... +
    # |   |                                                           |
    # +   +   +   +---+---+---+---+---+---+---+---+---+---+   +   +---+
    assert tuple(w[0, 0]) == (0, 1, 0, 0)
    assert tuple(w[0, 1]) == (0, 1, 0, 1)
    assert tuple(w[0, 3]) == (0, 0, 1, 1)
    assert tuple(w[0, 15]) == (0, 0, 1, 0)


def test_render_ascii_round_trip():
    w = mb.generate_maze(7, 9, seed=3, loop_fraction=0.15)
    text = mb.render_ascii(w, goal_cells=((3, 4),))
    assert " G " in text
    assert np.array_equal(mb.maze_to_wall_map(text), w)
