#!/usr/bin/env python3
"""完整的点云/TSDF重建流程示例。

该脚本把以下层串起来：
C. 预处理层
D. 全局配准层
E. 局部精配准层
F. 融合层
G. 网格重建层
H. 后处理层
I. 主流程控制层

依赖：
    pip install open3d numpy

示例：
    python pipeline_3d_reconstruction.py \
        --inputs data/frame1.pcd data/frame2.pcd data/frame3.pcd \
        --output-dir outputs

如果你已经有较好的初始位姿，可传入 --skip-global 来弱化/跳过全局配准。
如果传入 RGBD 数据目录与相机内参，也可以启用 TSDF 融合（见 --help）。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    import open3d as o3d
except ImportError as exc:  # 仅在实际运行算法时需要 open3d
    o3d = None
    OPEN3D_IMPORT_ERROR = exc
else:
    OPEN3D_IMPORT_ERROR = None


LOGGER = logging.getLogger("reconstruction_pipeline")


@dataclass
class PipelineConfig:
    voxel_size: float = 0.05
    normal_radius_factor: float = 2.0
    feature_radius_factor: float = 5.0
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    ransac_distance_factor: float = 1.5
    icp_distance_factor: float = 0.4
    tsdf_voxel_length: float = 0.01
    tsdf_sdf_trunc: float = 0.04
    tsdf_color_type: str = "RGB8"
    poisson_depth: int = 8
    mesh_smooth_iterations: int = 5
    skip_global: bool = False
    registration_mode: str = "icp"  # icp | gicp | colored_icp
    fusion_mode: str = "pointcloud"  # pointcloud | tsdf
    visualize: bool = False


@dataclass
class FrameRegistrationResult:
    source_path: str
    target_path: str
    transformation: List[List[float]]
    fitness: float
    inlier_rmse: float


def require_open3d() -> None:
    if np is None:
        raise RuntimeError(
            "缺少依赖 numpy。请先执行: pip install numpy open3d"
        ) from NUMPY_IMPORT_ERROR
    if o3d is None:
        raise RuntimeError(
            "缺少依赖 open3d。请先执行: pip install open3d numpy"
        ) from OPEN3D_IMPORT_ERROR


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# =========================
# C. 预处理层
# =========================
def voxel_downsample(pcd, voxel_size: float):
    LOGGER.debug("体素下采样: voxel_size=%s", voxel_size)
    return pcd.voxel_down_sample(voxel_size)



def remove_outliers(pcd, nb_neighbors: int, std_ratio: float):
    LOGGER.debug(
        "去离群点: nb_neighbors=%s, std_ratio=%s", nb_neighbors, std_ratio
    )
    filtered, indices = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return filtered.select_by_index(indices)



def estimate_normals(pcd, voxel_size: float, radius_factor: float):
    radius = voxel_size * radius_factor
    LOGGER.debug("法线估计: radius=%s", radius)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    pcd.normalize_normals()
    return pcd



def preprocess_point_cloud(pcd, config: PipelineConfig):
    LOGGER.info("开始预处理点云")
    pcd = remove_outliers(
        pcd,
        nb_neighbors=config.outlier_nb_neighbors,
        std_ratio=config.outlier_std_ratio,
    )
    pcd = voxel_downsample(pcd, config.voxel_size)
    pcd = estimate_normals(
        pcd,
        voxel_size=config.voxel_size,
        radius_factor=config.normal_radius_factor,
    )
    return pcd


# =========================
# D. 全局配准层
# =========================
def compute_features(pcd, config: PipelineConfig):
    radius = config.voxel_size * config.feature_radius_factor
    LOGGER.debug("计算 FPFH 特征: radius=%s", radius)
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100),
    )



def global_registration(src, tgt, src_feat, tgt_feat, config: PipelineConfig):
    distance_threshold = config.voxel_size * config.ransac_distance_factor
    LOGGER.info("执行全局配准 (RANSAC), distance_threshold=%s", distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src,
        tgt,
        src_feat,
        tgt_feat,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


# =========================
# E. 局部精配准层
# =========================
def run_icp(src, tgt, init_T: np.ndarray, config: PipelineConfig):
    threshold = config.voxel_size * config.icp_distance_factor
    LOGGER.info("执行点到面 ICP, threshold=%s", threshold)
    return o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        threshold,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )



def run_gicp(src, tgt, init_T: np.ndarray, config: PipelineConfig):
    threshold = config.voxel_size * config.icp_distance_factor
    LOGGER.info("执行 GICP, threshold=%s", threshold)
    return o3d.pipelines.registration.registration_generalized_icp(
        src,
        tgt,
        threshold,
        init_T,
    )



def run_colored_icp(src, tgt, init_T: np.ndarray, config: PipelineConfig):
    threshold = config.voxel_size * config.icp_distance_factor
    LOGGER.info("执行 Colored ICP, threshold=%s", threshold)
    return o3d.pipelines.registration.registration_colored_icp(
        src,
        tgt,
        threshold,
        init_T,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    )



def refine_registration(src, tgt, init_T: np.ndarray, config: PipelineConfig):
    mode = config.registration_mode.lower()
    if mode == "icp":
        return run_icp(src, tgt, init_T, config)
    if mode == "gicp":
        return run_gicp(src, tgt, init_T, config)
    if mode == "colored_icp":
        return run_colored_icp(src, tgt, init_T, config)
    raise ValueError(f"未知 registration_mode: {config.registration_mode}")


# =========================
# F. 融合层
# =========================
def fuse_point_cloud(global_map, new_pcd, pose: np.ndarray):
    transformed = copy.deepcopy(new_pcd)
    transformed.transform(pose)
    global_map += transformed
    return global_map



def build_intrinsics(width: int, height: int, intrinsic_json: Path):
    intrinsic_data = json.loads(intrinsic_json.read_text(encoding="utf-8"))
    return o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=intrinsic_data["fx"],
        fy=intrinsic_data["fy"],
        cx=intrinsic_data["cx"],
        cy=intrinsic_data["cy"],
    )



def integrate_tsdf(tsdf_volume, depth_path: Path, color_path: Path, pose: np.ndarray, intrinsics):
    color = o3d.io.read_image(str(color_path))
    depth = o3d.io.read_image(str(depth_path))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1000.0,
        depth_trunc=4.0,
        convert_rgb_to_intensity=False,
    )
    tsdf_volume.integrate(rgbd, intrinsics, np.linalg.inv(pose))


# =========================
# G. 网格重建层
# =========================
def extract_mesh_from_tsdf(tsdf_volume):
    LOGGER.info("从 TSDF 提取网格")
    mesh = tsdf_volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh



def poisson_reconstruction(pcd, depth: int):
    LOGGER.info("执行 Poisson 网格重建, depth=%s", depth)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
    )
    densities = np.asarray(densities)
    keep = densities > np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(~keep)
    mesh.compute_vertex_normals()
    return mesh


# =========================
# H. 后处理层
# =========================
def smooth_mesh(mesh, iterations: int):
    LOGGER.info("平滑网格, iterations=%s", iterations)
    smoothed = mesh.filter_smooth_taubin(number_of_iterations=iterations)
    smoothed.compute_vertex_normals()
    return smoothed



def remove_degenerate_faces(mesh):
    LOGGER.info("移除退化/重复/非流形元素")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh



def repair_holes(mesh):
    # Open3D 暂无强力的自动补洞接口，这里保留统一入口，便于未来替换其他库。
    LOGGER.info("repair_holes: 当前使用 Open3D，执行基础清理替代自动补洞")
    return remove_degenerate_faces(mesh)


# =========================
# I. 主流程控制层
# =========================
def load_point_cloud(path: Path):
    LOGGER.info("读取点云: %s", path)
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"点云为空或读取失败: {path}")
    return pcd



def save_transformation(path: Path, matrix: np.ndarray):
    path.write_text(json.dumps(matrix.tolist(), indent=2), encoding="utf-8")



def ensure_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir



def accumulate_poses(pairwise_transforms: Sequence[np.ndarray]) -> List[np.ndarray]:
    poses = [np.eye(4)]
    current = np.eye(4)
    for transform in pairwise_transforms:
        current = current @ transform
        poses.append(current.copy())
    return poses



def register_sequence(pcds: Sequence, paths: Sequence[Path], config: PipelineConfig):
    pairwise_transforms: List[np.ndarray] = []
    metrics: List[FrameRegistrationResult] = []

    for idx in range(1, len(pcds)):
        src = pcds[idx]
        tgt = pcds[idx - 1]
        src_path = paths[idx]
        tgt_path = paths[idx - 1]

        if config.skip_global:
            LOGGER.info("跳过全局配准，使用单位阵作为初值")
            init_T = np.eye(4)
        else:
            src_feat = compute_features(src, config)
            tgt_feat = compute_features(tgt, config)
            global_result = global_registration(src, tgt, src_feat, tgt_feat, config)
            init_T = global_result.transformation

        refined = refine_registration(src, tgt, init_T, config)
        pairwise_transforms.append(refined.transformation)
        metrics.append(
            FrameRegistrationResult(
                source_path=str(src_path),
                target_path=str(tgt_path),
                transformation=refined.transformation.tolist(),
                fitness=float(refined.fitness),
                inlier_rmse=float(refined.inlier_rmse),
            )
        )
        LOGGER.info(
            "完成配准: %s -> %s | fitness=%.4f | rmse=%.6f",
            src_path.name,
            tgt_path.name,
            refined.fitness,
            refined.inlier_rmse,
        )
    return accumulate_poses(pairwise_transforms), metrics



def run_pointcloud_fusion(pcds: Sequence, poses: Sequence[np.ndarray], config: PipelineConfig):
    LOGGER.info("执行点云融合")
    global_map = o3d.geometry.PointCloud()
    for pcd, pose in zip(pcds, poses):
        global_map = fuse_point_cloud(global_map, pcd, pose)
    global_map = voxel_downsample(global_map, config.voxel_size)
    global_map = estimate_normals(global_map, config.voxel_size, config.normal_radius_factor)
    return global_map



def run_tsdf_fusion(rgb_dir: Path, depth_dir: Path, poses: Sequence[np.ndarray], config: PipelineConfig, intrinsics_json: Path):
    LOGGER.info("执行 TSDF 融合")
    color_paths = sorted(rgb_dir.glob("*"))
    depth_paths = sorted(depth_dir.glob("*"))
    if len(color_paths) != len(depth_paths) or len(color_paths) != len(poses):
        raise ValueError("RGB/Depth/poses 数量不一致，无法进行 TSDF 融合")

    sample_color = o3d.io.read_image(str(color_paths[0]))
    width = np.asarray(sample_color).shape[1]
    height = np.asarray(sample_color).shape[0]
    intrinsics = build_intrinsics(width, height, intrinsics_json)

    color_type = getattr(
        o3d.pipelines.integration.TSDFVolumeColorType,
        config.tsdf_color_type,
    )
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config.tsdf_voxel_length,
        sdf_trunc=config.tsdf_sdf_trunc,
        color_type=color_type,
    )
    for color_path, depth_path, pose in zip(color_paths, depth_paths, poses):
        integrate_tsdf(tsdf_volume, depth_path, color_path, pose, intrinsics)
    return tsdf_volume



def save_outputs(output_dir: Path, poses: Sequence[np.ndarray], metrics: Sequence[FrameRegistrationResult], fused_pcd=None, mesh=None):
    output_dir = ensure_output_dir(output_dir)
    (output_dir / "poses.json").write_text(
        json.dumps([pose.tolist() for pose in poses], indent=2),
        encoding="utf-8",
    )
    (output_dir / "registration_metrics.json").write_text(
        json.dumps([asdict(item) for item in metrics], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if fused_pcd is not None:
        o3d.io.write_point_cloud(str(output_dir / "fused_map.ply"), fused_pcd)
    if mesh is not None:
        o3d.io.write_triangle_mesh(str(output_dir / "mesh.ply"), mesh)
    LOGGER.info("结果已保存到: %s", output_dir)



def maybe_visualize(geometries: Iterable, enabled: bool, window_name: str):
    if not enabled:
        return
    LOGGER.info("打开可视化窗口: %s", window_name)
    o3d.visualization.draw_geometries(list(geometries), window_name=window_name)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="完整的 3D 配准/融合/重建流水线")
    parser.add_argument("--inputs", nargs="+", help="输入点云文件列表，例如 *.pcd / *.ply")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="体素下采样大小")
    parser.add_argument("--skip-global", action="store_true", help="跳过全局配准")
    parser.add_argument(
        "--registration-mode",
        choices=["icp", "gicp", "colored_icp"],
        default="icp",
        help="局部精配准算法",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["pointcloud", "tsdf"],
        default="pointcloud",
        help="融合方式",
    )
    parser.add_argument("--rgb-dir", help="TSDF 模式下的 RGB 图像目录")
    parser.add_argument("--depth-dir", help="TSDF 模式下的深度图目录")
    parser.add_argument("--intrinsics-json", help="TSDF 模式下的相机内参 JSON 文件")
    parser.add_argument("--poisson-depth", type=int, default=8, help="Poisson 重建深度")
    parser.add_argument("--mesh-smooth-iterations", type=int, default=5, help="网格平滑迭代次数")
    parser.add_argument("--visualize", action="store_true", help="是否显示中间/最终结果")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    return parser.parse_args()



def validate_args(args: argparse.Namespace) -> None:
    if not args.inputs or len(args.inputs) < 2:
        raise ValueError("至少需要两个输入点云文件，才能进行配准与融合")
    if args.fusion_mode == "tsdf":
        required = [args.rgb_dir, args.depth_dir, args.intrinsics_json]
        if not all(required):
            raise ValueError("TSDF 模式需要同时提供 --rgb-dir --depth-dir --intrinsics-json")



def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        voxel_size=args.voxel_size,
        poisson_depth=args.poisson_depth,
        mesh_smooth_iterations=args.mesh_smooth_iterations,
        skip_global=args.skip_global,
        registration_mode=args.registration_mode,
        fusion_mode=args.fusion_mode,
        visualize=args.visualize,
    )



def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    require_open3d()
    validate_args(args)
    config = build_config(args)

    paths = [Path(item) for item in args.inputs]
    output_dir = Path(args.output_dir)

    raw_pcds = [load_point_cloud(path) for path in paths]
    processed_pcds = [preprocess_point_cloud(pcd, config) for pcd in raw_pcds]

    poses, metrics = register_sequence(processed_pcds, paths, config)

    fused_pcd = None
    mesh = None

    if config.fusion_mode == "pointcloud":
        fused_pcd = run_pointcloud_fusion(processed_pcds, poses, config)
        maybe_visualize([fused_pcd], config.visualize, "Fused Point Cloud")
        mesh = poisson_reconstruction(fused_pcd, config.poisson_depth)
    else:
        tsdf_volume = run_tsdf_fusion(
            rgb_dir=Path(args.rgb_dir),
            depth_dir=Path(args.depth_dir),
            poses=poses,
            config=config,
            intrinsics_json=Path(args.intrinsics_json),
        )
        mesh = extract_mesh_from_tsdf(tsdf_volume)

    mesh = smooth_mesh(mesh, config.mesh_smooth_iterations)
    mesh = remove_degenerate_faces(mesh)
    mesh = repair_holes(mesh)
    maybe_visualize([mesh], config.visualize, "Reconstructed Mesh")

    save_outputs(output_dir, poses, metrics, fused_pcd=fused_pcd, mesh=mesh)
    LOGGER.info("流水线执行完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
