import unittest

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

import semlaflow.util.functional as smolF


class TensorFnsTests(unittest.TestCase):
    def test_pairwise_concat_creates_stacked_pairs(self):
        vec_size = 4

        t = torch.rand((3, 2, vec_size))
        pairwise = smolF.pairwise_concat(t)

        expected_shape = (3, 2, 2, 2 * vec_size)
        first_vec = t[0, 0, :]
        second_vec = t[0, 1, :]

        self.assertEqual(expected_shape, pairwise.shape)

        self.assertTrue(torch.equal(first_vec, pairwise[0, 0, 0, :vec_size]))
        self.assertTrue(torch.equal(first_vec, pairwise[0, 0, 1, :vec_size]))

        self.assertTrue(torch.equal(second_vec, pairwise[0, 0, 1, vec_size:]))
        self.assertTrue(torch.equal(second_vec, pairwise[0, 1, 1, vec_size:]))

        self.assertTrue(torch.equal(first_vec, pairwise[0, 1, 0, vec_size:]))
        self.assertTrue(torch.equal(second_vec, pairwise[0, 1, 0, :vec_size]))

    def test_segment_sum_adds_feats_for_segments(self):
        batch_size = 2
        seq_len = 5
        num_feats = 4
        num_segments = 3

        t1 = torch.rand((seq_len, num_feats))
        t2 = torch.rand((seq_len, num_feats))
        data = torch.stack((t1, t2))
        segment_ids = torch.tensor([[0, 1, 1, 0, 2], [2, 2, 2, 0, 0]])

        expected_shape = (batch_size, num_segments, num_feats)

        exp_b0_s0 = t1[0] + t1[3]
        exp_b0_s1 = t1[1] + t1[2]
        exp_b0_s2 = t1[4]

        exp_b1_s0 = t2[3] + t2[4]
        exp_b1_s1 = torch.zeros(num_feats)
        exp_b1_s2 = t2[0] + t2[1] + t2[2]

        segment_sums = smolF.segment_sum(data, segment_ids, num_segments)

        self.assertEqual(expected_shape, segment_sums.shape)

        self.assertTrue(torch.equal(exp_b0_s0, segment_sums[0, 0]))
        self.assertTrue(torch.equal(exp_b0_s1, segment_sums[0, 1]))
        self.assertTrue(torch.equal(exp_b0_s2, segment_sums[0, 2]))

        self.assertTrue(torch.equal(exp_b1_s0, segment_sums[1, 0]))
        self.assertTrue(torch.equal(exp_b1_s1, segment_sums[1, 1]))
        self.assertTrue(torch.equal(exp_b1_s2, segment_sums[1, 2]))


class EdgeFnsTests(unittest.TestCase):
    def test_adj_from_node_mask_correct_adj(self):
        num_nodes = 4

        t1_nodes = torch.tensor([1, 1, 1, 1])
        t2_nodes = torch.tensor([1, 1, 1, 0])
        t3_nodes = torch.tensor([0, 0, 0, 0])
        node_mask = torch.stack((t1_nodes, t2_nodes, t3_nodes))

        exp_shape = (3, num_nodes, num_nodes)
        exp_type = torch.long

        b0_exp = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
        b1_exp = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
        b2_exp = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        adjacency = smolF.adj_from_node_mask(node_mask)

        self.assertEqual(exp_shape, adjacency.shape)
        self.assertEqual(exp_type, adjacency.dtype)

        self.assertEqual(b0_exp, adjacency[0].tolist())
        self.assertEqual(b1_exp, adjacency[1].tolist())
        self.assertEqual(b2_exp, adjacency[2].tolist())

    def test_edges_from_adj_correct_edges(self):
        t1 = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0], [2, -1, 0, 0]])
        t2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1]])
        adjacency = torch.stack((t1, t2))

        exp_shape = (2, 8)
        exp_type = torch.long

        exp_edges_i_b0 = [0, 0, 0, 0, 1, 1, 3, 3]
        exp_edges_j_b0 = [0, 1, 2, 3, 0, 2, 0, 1]

        exp_edges_i_b1 = [1, 3, 0, 0, 0, 0, 0, 0]
        exp_edges_j_b1 = [3, 3, 0, 0, 0, 0, 0, 0]

        exp_mask_b0 = [1, 1, 1, 1, 1, 1, 1, 1]
        exp_mask_b1 = [1, 1, 0, 0, 0, 0, 0, 0]

        (edge_is, edge_js), edge_mask = smolF.edges_from_adj(adjacency)

        # Check shapes
        self.assertEqual(exp_shape, edge_is.shape)
        self.assertEqual(exp_shape, edge_js.shape)
        self.assertEqual(exp_shape, edge_mask.shape)

        # Check types
        self.assertEqual(exp_type, edge_is.dtype)
        self.assertEqual(exp_type, edge_js.dtype)
        self.assertEqual(exp_type, edge_mask.dtype)

        # Check edge indices
        self.assertEqual(exp_edges_i_b0, edge_is[0].tolist())
        self.assertEqual(exp_edges_j_b0, edge_js[0].tolist())
        self.assertEqual(exp_edges_i_b1, edge_is[1].tolist())
        self.assertEqual(exp_edges_j_b1, edge_js[1].tolist())

        # Check mask
        self.assertEqual(exp_mask_b0, edge_mask[0].tolist())
        self.assertEqual(exp_mask_b1, edge_mask[1].tolist())

    def test_adj_from_edges_correct_adj(self):
        num_nodes = 4

        edges = torch.tensor([[0, 0, 1], [0, 2, 2], [1, 0, 1], [1, 3, 0], [2, 2, 3], [3, 1, 1]])

        exp_shape = (num_nodes, num_nodes)
        exp_type = torch.long

        exp_adj = [[1, 0, 2, 0], [1, 0, 0, 0], [0, 0, 3, 0], [0, 1, 0, 0]]

        edge_indices = edges[:, :2]
        edge_types = edges[:, 2]

        adjacency = smolF.adj_from_edges(edge_indices, edge_types, num_nodes)

        self.assertEqual(exp_shape, adjacency.shape)
        self.assertEqual(exp_type, adjacency.dtype)

        self.assertEqual(exp_adj, adjacency.tolist())

    def test_edges_from_nodes_fully_connected(self):
        num_nodes = 4

        coords_b0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 3.0, 1.0], [4.0, -2.0, -3.0]])
        coords_b1 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [-1.0, -5.0, 2.0]])
        coords = torch.stack((coords_b0, coords_b1))

        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

        exp_shape = (2, num_nodes, num_nodes)
        exp_type = torch.long

        exp_adj_b0 = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
        exp_adj_b1 = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        adjacency = smolF.edges_from_nodes(coords, node_mask=mask)

        self.assertEqual(exp_shape, adjacency.shape)
        self.assertEqual(exp_type, adjacency.dtype)

        self.assertEqual(exp_adj_b0, adjacency[0].tolist())
        self.assertEqual(exp_adj_b1, adjacency[1].tolist())

    def test_edges_from_nodes_correct_neighbours(self):
        num_nodes = 4

        coords_b0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 3.0, 1.0], [4.0, -2.0, -3.0]])
        coords_b1 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [-1.0, -5.0, 2.0]])
        coords = torch.stack((coords_b0, coords_b1))

        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

        exp_shape = (2, num_nodes, num_nodes)
        exp_type = torch.long

        exp_adj_b0 = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0]]
        exp_adj_b1 = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        adjacency = smolF.edges_from_nodes(coords, k=2, node_mask=mask)

        self.assertEqual(exp_shape, adjacency.shape)
        self.assertEqual(exp_type, adjacency.dtype)

        self.assertEqual(exp_adj_b0, adjacency[0].tolist())
        self.assertEqual(exp_adj_b1, adjacency[1].tolist())


class SparseFnsTests(unittest.TestCase):
    def test_gather_edge_features(self):
        feats_b0 = torch.tensor(
            [
                [[0.5, 1.0], [0.1, -0.5], [5.0, -2.0], [-0.1, 0.8]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
                [[0.6, -0.2], [0.5, -2.0], [-7.0, 4.0], [5.0, 6.0]],
            ]
        )
        feats_b1 = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                [[-1.0, -2.0], [-2.0, -3.0], [-3.0, -4.0], [-4.0, -5.0]],
                [[0.6, 0.9], [0.3, 0.2], [0.1, -0.7], [-0.5, 0.9]],
                [[1.5, -2.8], [6.3, 2.9], [5.8, 9.1], [0.4, -3.7]],
            ]
        )
        feats = torch.stack((feats_b0, feats_b1))

        adj_1 = torch.tensor(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
            ]
        ).long()

        adj_2 = torch.tensor(
            [
                [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0]],
                [[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
            ]
        ).long()

        exp_feats_1_b0 = [[[0.5, 1.0]], [[3.0, 4.0]], [[13.0, 14.0]], [[5.0, 6.0]]]
        exp_feats_1_b1 = [[[0.7, 0.8]], [[-4.0, -5.0]], [[-0.5, 0.9]], [[0.4, -3.7]]]
        exp_feats_1 = [exp_feats_1_b0, exp_feats_1_b1]

        exp_feats_2_b0 = [
            [[0.1, -0.5], [5.0, -2.0]],
            [[1.0, 2.0], [7.0, 8.0]],
            [[11.0, 12.0], [15.0, 16.0]],
            [[0.6, -0.2], [0.5, -2.0]],
        ]
        exp_feats_2_b1 = [
            [[0.3, 0.4], [0.7, 0.8]],
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[0.6, 0.9], [0.3, 0.2]],
            [[6.3, 2.9], [5.8, 9.1]],
        ]
        exp_feats_2 = [exp_feats_2_b0, exp_feats_2_b1]

        gathered_1 = smolF.gather_edge_features(feats, adj_1)
        gathered_2 = smolF.gather_edge_features(feats, adj_2)

        np.testing.assert_almost_equal(exp_feats_1, gathered_1.tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_feats_2, gathered_2.tolist(), decimal=5)


class GeometryFnsTests(unittest.TestCase):
    def test_calc_distance_without_edges(self):
        num_nodes = 4

        coords_b0 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [5.0, -1.0, -2.0]])
        coords_b1 = torch.tensor([[0.5, 1.0, -0.25], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        coords = torch.stack((coords_b0, coords_b1))

        exp_shape = (2, num_nodes, num_nodes)
        exp_type = torch.float

        exp_b0 = [[0.0, 0.0, 5.0, 29.0], [0.0, 0.0, 5.0, 29.0], [5.0, 5.0, 0.0, 46.0], [29.0, 29.0, 46.0, 0.0]]
        exp_b1 = [
            [0.0, 1.8125, 1.3125, 1.3125],
            [1.8125, 0.0, 3.0, 3.0],
            [1.3125, 3.0, 0.0, 0.0],
            [1.3125, 3.0, 0.0, 0.0],
        ]

        sqrd_dists = smolF.calc_distances(coords, sqrd=True)
        dists = torch.sqrt(sqrd_dists)

        self.assertEqual(exp_shape, sqrd_dists.shape)
        self.assertEqual(exp_type, sqrd_dists.dtype)

        np.testing.assert_almost_equal(exp_b0, sqrd_dists[0].tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_b1, sqrd_dists[1].tolist(), decimal=5)

        np.testing.assert_almost_equal(np.sqrt(exp_b0).tolist(), dists[0].tolist(), decimal=5)
        np.testing.assert_almost_equal(np.sqrt(exp_b1).tolist(), dists[1].tolist(), decimal=5)

    def test_calc_distances_from_edges(self):
        num_edges = 8

        coords_b0 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [5.0, -1.0, -2.0]])
        coords_b1 = torch.tensor([[0.5, 1.0, -0.25], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        coords = torch.stack((coords_b0, coords_b1))

        edge_is = torch.tensor([[0, 0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 1, 1, 0, 0, 0]])
        edge_js = torch.tensor([[0, 1, 2, 3, 0, 2, 0, 0], [0, 1, 2, 2, 2, 0, 0, 0]])
        edges = (edge_is, edge_js)

        exp_shape = (2, num_edges)
        exp_type = torch.float

        exp_b0 = [0.0, 0.0, 5.0, 29.0, 0.0, 0.0, 5.0, 5.0]
        exp_b1 = [0.0, 1.8125, 1.3125, 3.0, 3.0, 0.0, 0.0, 0.0]

        sqrd_dists = smolF.calc_distances(coords, edges=edges, sqrd=True)
        dists = torch.sqrt(sqrd_dists)

        self.assertEqual(exp_shape, sqrd_dists.shape)
        self.assertEqual(exp_type, sqrd_dists.dtype)

        np.testing.assert_almost_equal(exp_b0, sqrd_dists[0].tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_b1, sqrd_dists[1].tolist(), decimal=5)

        np.testing.assert_almost_equal(np.sqrt(exp_b0).tolist(), dists[0].tolist(), decimal=5)
        np.testing.assert_almost_equal(np.sqrt(exp_b1).tolist(), dists[1].tolist(), decimal=5)

    def test_calc_com_correct_centre(self):
        coords_b0 = torch.tensor([[1.0, 1.0, 1.0], [2.0, -2.0, 0.0], [-4.0, 2.0, 2.0], [3.0, -5.0, -5.0]])
        coords_b1 = torch.tensor([[1.0, 1.0, 1.0], [2.0, -2.0, 0.0], [-4.0, 2.0, 1.0], [3.0, -5.0, -5.0]])
        coords_b2 = torch.tensor([[1.0, 1.0, 1.0], [2.0, -2.0, 0.0], [-4.0, 2.0, 1.0], [3.0, -5.0, -5.0]])
        coords = torch.stack((coords_b0, coords_b1, coords_b2))
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]])

        exp_shape = (3, 1, 3)
        exp_type = torch.float

        exp_com_b0 = [0.5, -1.0, -0.5]
        exp_com_b1 = [1.5, -0.5, 0.5]
        exp_com_b2 = [np.nan, np.nan, np.nan]

        com = smolF.calc_com(coords, node_mask=mask)

        self.assertEqual(exp_shape, com.shape)
        self.assertEqual(exp_type, com.dtype)

        self.assertEqual(exp_com_b0, com[0, 0, :].tolist())
        self.assertEqual(exp_com_b1, com[1, 0, :].tolist())
        np.testing.assert_equal(exp_com_b2, com[2, 0, :].tolist())

    def test_rotate_rotates_all_coords_correctly(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 2.0, 0.5]])

        rot1 = [np.pi / 2, 0.0, 0.0]
        rot2 = [0.0, np.pi / 2, 0.0]
        rot3 = [0.0, 0.0, np.pi / 2]
        rot4 = [np.pi / 2, np.pi, np.pi / 2]
        rot5 = [-np.pi / 2, 0.0, 2 * np.pi]

        exp_coords_1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [-1.0, -0.5, 2.0]]
        exp_coords_2 = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 2.0, 1.0]]
        exp_coords_3 = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-2.0, -1.0, 0.5]]
        exp_coords_4 = [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.5, 1.0, -2.0]]
        exp_coords_5 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.5, -2.0]]

        rotated_1 = smolF.rotate(coords, rot1)
        rotated_2 = smolF.rotate(coords, rot2)
        rotated_3 = smolF.rotate(coords, rot3)
        rotated_4 = smolF.rotate(coords, rot4)
        rotated_5 = smolF.rotate(coords, rot5)

        np.testing.assert_almost_equal(exp_coords_1, rotated_1.tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_coords_2, rotated_2.tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_coords_3, rotated_3.tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_coords_4, rotated_4.tolist(), decimal=5)
        np.testing.assert_almost_equal(exp_coords_5, rotated_5.tolist(), decimal=5)

    def test_rotate_agrees_with_scipy(self):
        coords = torch.rand((10, 3))

        angles_1 = (np.random.rand(3) * np.pi * 2).tolist()
        angles_2 = (np.random.rand(3) * np.pi * 2).tolist()
        angles_3 = (np.random.rand(3) * np.pi * 2).tolist()

        rot1 = Rotation.from_euler("xyz", angles_1)
        rot2 = Rotation.from_euler("xyz", angles_2)
        rot3 = Rotation.from_euler("xyz", angles_3)

        exp_coords_1 = rot1.apply(coords.tolist())
        exp_coords_2 = rot2.apply(coords.tolist())
        exp_coords_3 = rot3.apply(coords.tolist())

        rotated_1 = smolF.rotate(coords, angles_1)
        rotated_2 = smolF.rotate(coords, angles_2)
        rotated_3 = smolF.rotate(coords, angles_3)

        np.testing.assert_almost_equal(exp_coords_1, rotated_1, decimal=5)
        np.testing.assert_almost_equal(exp_coords_2, rotated_2, decimal=5)
        np.testing.assert_almost_equal(exp_coords_3, rotated_3, decimal=5)


class SinkhornFnsTests(unittest.TestCase):
    def test_sinkhorn_rows_and_cols_sum_to_one(self):
        torch.manual_seed(0)
        coords1 = torch.rand((6, 3))
        coords2 = torch.rand((6, 3))
        cost_matrix = smolF.inter_distances(coords1, coords2, sqrd=True)

        plan = smolF.sinkhorn(cost_matrix, eps=0.1, n_iters=200)

        row_sums = plan.sum(dim=1)
        col_sums = plan.sum(dim=0)

        np.testing.assert_almost_equal(torch.ones(6).tolist(), row_sums.tolist(), decimal=4)
        np.testing.assert_almost_equal(torch.ones(6).tolist(), col_sums.tolist(), decimal=4)

    def test_sinkhorn_converges_to_hard_assignment_as_eps_shrinks(self):
        torch.manual_seed(1)
        coords1 = torch.rand((5, 3))
        coords2 = torch.rand((5, 3))
        cost_matrix = smolF.inter_distances(coords1, coords2, sqrd=True)

        row_indices, col_indices = linear_sum_assignment(cost_matrix.numpy())
        exp_hard_plan = np.zeros((5, 5))
        exp_hard_plan[row_indices, col_indices] = 1.0

        plan = smolF.sinkhorn(cost_matrix, eps=1e-4, n_iters=1000)

        np.testing.assert_almost_equal(exp_hard_plan, plan.numpy(), decimal=2)

    def test_sinkhorn_uniform_as_eps_grows(self):
        torch.manual_seed(2)
        coords1 = torch.rand((4, 3))
        coords2 = torch.rand((4, 3))
        cost_matrix = smolF.inter_distances(coords1, coords2, sqrd=True)

        plan = smolF.sinkhorn(cost_matrix, eps=1e5, n_iters=200)

        exp_uniform = torch.full((4, 4), 0.25)
        np.testing.assert_almost_equal(exp_uniform.tolist(), plan.tolist(), decimal=3)

    def test_sinkhorn_stable_for_very_small_eps(self):
        torch.manual_seed(3)
        coords1 = torch.rand((8, 3))
        coords2 = torch.rand((8, 3))
        cost_matrix = smolF.inter_distances(coords1, coords2, sqrd=True)

        plan = smolF.sinkhorn(cost_matrix, eps=1e-6, n_iters=200)

        self.assertFalse(torch.isnan(plan).any().item())
        self.assertFalse(torch.isinf(plan).any().item())

    def test_sinkhorn_rejects_non_square_cost_matrix(self):
        cost_matrix = torch.rand((3, 4))
        with self.assertRaises(ValueError):
            smolF.sinkhorn(cost_matrix, eps=0.1)

    def test_sinkhorn_rejects_non_positive_eps(self):
        cost_matrix = torch.rand((3, 3))
        with self.assertRaises(ValueError):
            smolF.sinkhorn(cost_matrix, eps=0.0)


def _hungarian_init(cost, seq_lengths):
    batch_size, n, _ = cost.shape
    perm = torch.arange(n).unsqueeze(0).repeat(batch_size, 1).clone()
    for b in range(batch_size):
        n_b = seq_lengths[b].item()
        _, col_ind = linear_sum_assignment(cost[b, :n_b, :n_b].numpy())
        perm[b, :n_b] = torch.as_tensor(col_ind)
    return perm


def _perm_cost(cost, perm, seq_lengths):
    total = 0.0
    for b in range(cost.size(0)):
        n_b = seq_lengths[b].item()
        for i in range(n_b):
            total += cost[b, i, perm[b, i]].item()
    return total


class MCMCPermutationFnsTests(unittest.TestCase):
    def _random_batch(self, seq_lengths, seed):
        torch.manual_seed(seed)
        batch_size = len(seq_lengths)
        n = max(seq_lengths)
        seq_lengths_t = torch.tensor(seq_lengths)
        to_coords = torch.rand((batch_size, n, 3))
        from_coords = torch.rand((batch_size, n, 3))
        node_mask = (torch.arange(n).unsqueeze(0) < seq_lengths_t.unsqueeze(1)).long()
        cost = smolF.inter_distances(to_coords, from_coords, sqrd=True)
        return cost, node_mask, to_coords, seq_lengths_t

    def test_padded_positions_stay_at_init_value(self):
        cost, node_mask, to_coords, seq_lengths = self._random_batch([5, 3, 5], seed=0)
        init_perm = _hungarian_init(cost, seq_lengths)

        out = smolF.mcmc_permutation(
            cost, node_mask, eps=torch.full((3,), 0.5), n_iters=100,
            init_perm=init_perm.clone(), proposal="uniform",
        )

        self.assertTrue(torch.equal(out[1, 3:], init_perm[1, 3:]))

    def test_tiny_eps_stays_at_already_optimal_init(self):
        cost, node_mask, to_coords, seq_lengths = self._random_batch([6, 6], seed=1)
        init_perm = _hungarian_init(cost, seq_lengths)

        out = smolF.mcmc_permutation(
            cost, node_mask, eps=torch.full((2,), 1e-4), n_iters=200,
            init_perm=init_perm.clone(), proposal="knn", knn_k=4, to_coords=to_coords,
        )

        self.assertTrue(torch.equal(out, init_perm))

    def test_cost_does_not_increase_on_average_from_hungarian_init(self):
        for proposal in ["uniform", "knn"]:
            costs_after = []
            for seed in range(5):
                cost, node_mask, to_coords, seq_lengths = self._random_batch([8, 8], seed=seed)
                init_perm = _hungarian_init(cost, seq_lengths)
                init_cost = _perm_cost(cost, init_perm, seq_lengths)

                out = smolF.mcmc_permutation(
                    cost, node_mask, eps=torch.full((2,), 0.02), n_iters=100,
                    init_perm=init_perm.clone(), proposal=proposal, knn_k=4, to_coords=to_coords,
                )
                costs_after.append(_perm_cost(cost, out, seq_lengths) - init_cost)

            self.assertLessEqual(np.mean(costs_after), 0.5)

    def test_n_less_than_two_short_circuits(self):
        node_mask = torch.ones((2, 1), dtype=torch.long)
        cost = torch.zeros((2, 1, 1))
        init_perm = torch.zeros((2, 1), dtype=torch.long)

        out = smolF.mcmc_permutation(
            cost, node_mask, eps=torch.ones(2), n_iters=10, init_perm=init_perm, proposal="uniform"
        )

        self.assertTrue(torch.equal(out, init_perm))

    def test_mixed_batch_with_single_atom_molecule_does_not_crash_or_move(self):
        cost, node_mask, to_coords, seq_lengths = self._random_batch([1, 6], seed=2)
        init_perm = _hungarian_init(cost, seq_lengths)

        for proposal in ["uniform", "knn"]:
            out = smolF.mcmc_permutation(
                cost, node_mask, eps=torch.full((2,), 0.5), n_iters=50,
                init_perm=init_perm.clone(), proposal=proposal, knn_k=8, to_coords=to_coords,
            )
            self.assertTrue(torch.equal(out[0, :1], init_perm[0, :1]))

    def test_knn_k_larger_than_n_minus_one_does_not_crash(self):
        cost, node_mask, to_coords, seq_lengths = self._random_batch([4, 4], seed=3)
        init_perm = _hungarian_init(cost, seq_lengths)

        out = smolF.mcmc_permutation(
            cost, node_mask, eps=torch.full((2,), 0.5), n_iters=20,
            init_perm=init_perm.clone(), proposal="knn", knn_k=100, to_coords=to_coords,
        )
        self.assertEqual(out.shape, init_perm.shape)

    def test_result_stays_a_valid_permutation_per_molecule(self):
        cost, node_mask, to_coords, seq_lengths = self._random_batch([5, 3, 7], seed=4)
        init_perm = _hungarian_init(cost, seq_lengths)

        out = smolF.mcmc_permutation(
            cost, node_mask, eps=torch.full((3,), 0.3), n_iters=100,
            init_perm=init_perm.clone(), proposal="knn", knn_k=3, to_coords=to_coords,
        )

        for b in range(3):
            n_b = seq_lengths[b].item()
            self.assertEqual(sorted(out[b, :n_b].tolist()), list(range(n_b)))

    def test_rejects_unknown_proposal(self):
        cost, node_mask, to_coords, seq_lengths = self._random_batch([4, 4], seed=5)
        init_perm = _hungarian_init(cost, seq_lengths)
        with self.assertRaises(ValueError):
            smolF.mcmc_permutation(cost, node_mask, eps=torch.ones(2), n_iters=5, init_perm=init_perm, proposal="bogus")

    def test_knn_proposal_requires_to_coords(self):
        cost, node_mask, to_coords, seq_lengths = self._random_batch([4, 4], seed=6)
        init_perm = _hungarian_init(cost, seq_lengths)
        with self.assertRaises(ValueError):
            smolF.mcmc_permutation(cost, node_mask, eps=torch.ones(2), n_iters=5, init_perm=init_perm, proposal="knn")


class SinkhornBatchedFnsTests(unittest.TestCase):
    def _random_batch(self, seq_lengths, seed):
        torch.manual_seed(seed)
        batch_size = len(seq_lengths)
        n = max(seq_lengths)
        seq_lengths_t = torch.tensor(seq_lengths)
        to_coords = torch.rand((batch_size, n, 3))
        from_coords = torch.rand((batch_size, n, 3))
        node_mask = (torch.arange(n).unsqueeze(0) < seq_lengths_t.unsqueeze(1)).long()
        cost = smolF.inter_distances(to_coords, from_coords, sqrd=True)
        return cost, node_mask, seq_lengths_t

    def test_real_submatrix_rows_and_cols_sum_to_one(self):
        cost, node_mask, seq_lengths = self._random_batch([5, 3, 6], seed=0)
        eps = torch.full((3,), 0.1)

        plan = smolF.sinkhorn_batched(cost, node_mask, eps, n_iters=1000)

        for b in range(3):
            n_b = seq_lengths[b].item()
            sub = plan[b, :n_b, :n_b]
            np.testing.assert_almost_equal(torch.ones(n_b).tolist(), sub.sum(dim=1).tolist(), decimal=2)
            np.testing.assert_almost_equal(torch.ones(n_b).tolist(), sub.sum(dim=0).tolist(), decimal=2)

    def test_matches_unbatched_sinkhorn_when_no_padding(self):
        cost, node_mask, seq_lengths = self._random_batch([5, 5], seed=1)
        eps = torch.full((2,), 0.2)

        batched = smolF.sinkhorn_batched(cost, node_mask, eps, n_iters=200)
        for b in range(2):
            unbatched = smolF.sinkhorn(cost[b], eps[b].item(), n_iters=200)
            np.testing.assert_almost_equal(unbatched.tolist(), batched[b].tolist(), decimal=4)

    def test_converges_to_hungarian_as_eps_shrinks(self):
        cost, node_mask, seq_lengths = self._random_batch([5, 5], seed=2)
        eps = torch.full((2,), 1e-3)  # matches COUPLING_MIN_EPS, the real floor used in training

        plan = smolF.sinkhorn_batched(cost, node_mask, eps, n_iters=3000)

        for b in range(2):
            n_b = seq_lengths[b].item()
            row_ind, col_ind = linear_sum_assignment(cost[b, :n_b, :n_b].numpy())
            exp_hard_plan = np.zeros((n_b, n_b))
            exp_hard_plan[row_ind, col_ind] = 1.0
            np.testing.assert_almost_equal(exp_hard_plan, plan[b, :n_b, :n_b].numpy(), decimal=2)

    def test_padding_does_not_leak_into_real_submatrix(self):
        # Same molecule (n_b=4) once alone (no padding) and once inside a batch padded up to n=7
        torch.manual_seed(3)
        to_coords = torch.rand((4, 3))
        from_coords = torch.rand((4, 3))
        cost_small = smolF.inter_distances(to_coords, from_coords, sqrd=True)

        cost_padded = torch.zeros((1, 7, 7))
        cost_padded[0, :4, :4] = cost_small
        node_mask = torch.zeros((1, 7), dtype=torch.long)
        node_mask[0, :4] = 1

        eps = torch.full((1,), 0.15)
        plan_padded = smolF.sinkhorn_batched(cost_padded, node_mask, eps, n_iters=1000)
        plan_small = smolF.sinkhorn_batched(
            cost_small.unsqueeze(0), torch.ones((1, 4), dtype=torch.long), eps, n_iters=1000
        )

        np.testing.assert_almost_equal(
            plan_small[0].numpy(), plan_padded[0, :4, :4].numpy(), decimal=2
        )

    def test_rejects_non_positive_eps(self):
        cost, node_mask, seq_lengths = self._random_batch([4, 4], seed=4)
        with self.assertRaises(ValueError):
            smolF.sinkhorn_batched(cost, node_mask, eps=torch.tensor([0.1, 0.0]))

    def test_rejects_mismatched_cost_shape(self):
        cost, node_mask, seq_lengths = self._random_batch([4, 4], seed=5)
        with self.assertRaises(ValueError):
            smolF.sinkhorn_batched(cost[:, :, :-1], node_mask, eps=torch.ones(2))


class PlanFromSinkhornFnsTests(unittest.TestCase):
    def _random_plan(self, seq_lengths, seed, n_iters=1000, eps_val=0.1):
        torch.manual_seed(seed)
        batch_size = len(seq_lengths)
        n = max(seq_lengths)
        seq_lengths_t = torch.tensor(seq_lengths)
        to_coords = torch.rand((batch_size, n, 3))
        from_coords = torch.rand((batch_size, n, 3))
        node_mask = (torch.arange(n).unsqueeze(0) < seq_lengths_t.unsqueeze(1)).long()
        cost = smolF.inter_distances(to_coords, from_coords, sqrd=True)
        eps = torch.full((batch_size,), eps_val)
        raw = smolF.sinkhorn_batched(cost, node_mask, eps, n_iters=n_iters)
        return raw, node_mask, seq_lengths_t

    def test_padding_rows_become_identity(self):
        raw, node_mask, seq_lengths = self._random_plan([5, 3, 6], seed=0)

        plan, _ = smolF.plan_from_sinkhorn(raw, node_mask)

        for b in range(3):
            for i in range(seq_lengths[b].item(), node_mask.size(1)):
                self.assertEqual(plan[b, i, i].item(), 1.0)
                self.assertEqual(plan[b, i, :].sum().item(), 1.0)

    def test_cross_terms_between_real_and_padding_are_zero(self):
        raw, node_mask, seq_lengths = self._random_plan([5, 3, 6], seed=1)

        plan, _ = smolF.plan_from_sinkhorn(raw, node_mask)

        n = node_mask.size(1)
        for b in range(3):
            n_b = seq_lengths[b].item()
            self.assertEqual(plan[b, :n_b, n_b:].abs().sum().item(), 0.0)
            self.assertEqual(plan[b, n_b:, :n_b].abs().sum().item(), 0.0)

    def test_rows_sum_to_one_even_when_unconverged(self):
        # 5 iterations is deliberately nowhere near converged, which is the regime a training step
        # actually runs in -- the row sums are the marginal sinkhorn_batched leaves approximate.
        raw, node_mask, seq_lengths = self._random_plan([5, 3, 6], seed=2, n_iters=5, eps_val=0.01)

        plan, row_dev = smolF.plan_from_sinkhorn(raw, node_mask)

        self.assertGreater(row_dev.max().item(), 1e-3)
        np.testing.assert_almost_equal(
            plan.sum(dim=2).numpy(), torch.ones_like(plan.sum(dim=2)).numpy(), decimal=6
        )

    def test_row_deviation_is_small_once_converged(self):
        # The floor here is float32 precision, not iteration count -- 2000 iters gives ~5e-4 and
        # 5000 gives ~2e-4. At the 100 iters a training step actually affords it is ~1e-2, which
        # is why the renormalisation above is load-bearing rather than cosmetic.
        raw, node_mask, seq_lengths = self._random_plan([5, 3, 6], seed=3, n_iters=2000)

        _, row_dev = smolF.plan_from_sinkhorn(raw, node_mask)

        self.assertLess(row_dev.max().item(), 1e-3)

    def test_renormalise_disabled_leaves_rows_unnormalised(self):
        raw, node_mask, seq_lengths = self._random_plan([5, 3], seed=4, n_iters=5, eps_val=0.01)

        plan, row_dev = smolF.plan_from_sinkhorn(raw, node_mask, renormalise=False)

        np.testing.assert_almost_equal(
            (plan.sum(dim=2) - 1.0).abs().amax(dim=1).numpy(), row_dev.numpy(), decimal=6
        )

    def test_rejects_mismatched_plan_shape(self):
        raw, node_mask, seq_lengths = self._random_plan([4, 4], seed=5)
        with self.assertRaises(ValueError):
            smolF.plan_from_sinkhorn(raw[:, :, :-1], node_mask)


class PermutationToPlanFnsTests(unittest.TestCase):
    def _perm(self, seq_lengths, seed):
        torch.manual_seed(seed)
        n = max(seq_lengths)
        seq_lengths_t = torch.tensor(seq_lengths)
        node_mask = (torch.arange(n).unsqueeze(0) < seq_lengths_t.unsqueeze(1)).long()
        perm = torch.arange(n).unsqueeze(0).repeat(len(seq_lengths), 1).clone()
        for b, n_b in enumerate(seq_lengths):
            perm[b, :n_b] = torch.randperm(n_b)
        return perm, node_mask, seq_lengths_t

    def test_plan_applied_to_a_tensor_matches_a_gather(self):
        perm, node_mask, seq_lengths = self._perm([5, 3, 6], seed=0)
        x = torch.rand((3, node_mask.size(1), 4))

        plan = smolF.permutation_to_plan(perm, node_mask)

        np.testing.assert_almost_equal(
            (plan @ x).numpy(), torch.gather(x, 1, perm.unsqueeze(2).expand(-1, -1, 4)).numpy(), decimal=6
        )

    def test_rows_sum_to_one_and_padding_is_identity(self):
        perm, node_mask, seq_lengths = self._perm([5, 3, 6], seed=1)

        plan = smolF.permutation_to_plan(perm, node_mask)

        np.testing.assert_almost_equal(
            plan.sum(dim=2).numpy(), torch.ones_like(plan.sum(dim=2)).numpy(), decimal=6
        )
        for b in range(3):
            for i in range(seq_lengths[b].item(), node_mask.size(1)):
                self.assertEqual(plan[b, i, i].item(), 1.0)

    def test_identity_permutation_gives_the_identity_matrix(self):
        n = 5
        node_mask = torch.ones((2, n), dtype=torch.long)
        perm = torch.arange(n).unsqueeze(0).repeat(2, 1)

        plan = smolF.permutation_to_plan(perm, node_mask)

        np.testing.assert_almost_equal(plan.numpy(), torch.eye(n).expand(2, n, n).numpy(), decimal=6)

    def test_rejects_mismatched_perm_shape(self):
        perm, node_mask, seq_lengths = self._perm([4, 4], seed=2)
        with self.assertRaises(ValueError):
            smolF.permutation_to_plan(perm[:, :-1], node_mask)


if __name__ == "__main__":
    unittest.main()
